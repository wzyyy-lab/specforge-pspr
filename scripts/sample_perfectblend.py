"""Stratified sample of open-perfectblend into the SpecForge prompt jsonl format.

``scripts/prepare_data.py --dataset perfectblend`` pulls from the HF hub, but this host's hub cache
for ``mlabonne/open-perfectblend`` is an empty stub (only ``refs/main``). The full snapshot is on
disk as parquet, so read it directly and keep the emitted rows byte-compatible with what
``process_sharegpt_row`` would have produced -- ``{"id", "conversations": [{"role", "content"}]}``
with ``ROLE_MAPPING`` applied -- so ``prepare_prompt_tasks`` treats the file exactly like any other
prepared dataset.

Sampling is proportional per ``source``: each of the 8 sources contributes its population share of
the requested total, drawn with a fixed seed inside the group. A uniform sample over the whole file
would give the same expectation but not the same realisation, and the point of the stratification is
that the realised per-source mix is exactly the population mix rather than approximately.

Rows whose conversation carries no assistant turn are excluded from the population before quotas are
computed, not filtered after. ``HuggingFaceH4/ultrafeedback_binarized`` is a preference set and ~37%
of its rows keep only the prompt (measured; every other source is at 100%), so those rows have an
all-zero loss mask and would be dropped later by ``min_loss_tokens`` anyway. Apportioning over raw
row counts and filtering afterwards therefore under-fills that one source by exactly its share of
unusable rows -- the first run of this script missed its quota by 6540 rows that way. Proportional
here means proportional over *trainable* rows.

Run::

    python scripts/sample_perfectblend.py \
        --source-dir /home/wangzhuoyu/sp-decoding-iclr/data/open-perfectblend/data \
        --total 200000 --output cache/dataset/perfectblend_200k.jsonl
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import sys
from pathlib import Path

import pyarrow.parquet as pq

ROLE_MAPPING = {
    "human": "user",
    "gpt": "assistant",
    "chatgpt": "assistant",
    "bing": "assistant",
    "bard": "assistant",
}


def role_mapped(conversation: list) -> tuple[list[dict], int]:
    """Apply ``ROLE_MAPPING``; returns ``(messages, n_unmapped)``."""
    messages, unmapped = [], 0
    for message in conversation:
        role = ROLE_MAPPING.get(message["from"])
        if role is None:
            unmapped += 1
            continue
        messages.append({"role": role, "content": message["value"]})
    return messages, unmapped


def trainable(messages: list[dict]) -> bool:
    """A row is only usable if some turn is supervised, i.e. it has an assistant reply."""
    return any(m["role"] == "assistant" for m in messages)


def population(files: list[Path]) -> tuple[dict[str, list[tuple[int, int]]], int, int, dict]:
    """Return ``({source: [(file_idx, row_idx)]}, n_trainable, n_scanned, unmapped_roles)``.

    Only trainable rows enter the returned groups, so quotas are apportioned over what can actually
    be trained on rather than over raw row counts.
    """
    groups: dict[str, list[tuple[int, int]]] = collections.defaultdict(list)
    unmapped_roles: dict[str, int] = collections.Counter()
    scanned = 0
    for file_index, path in enumerate(files):
        parquet = pq.ParquetFile(path)
        row_base = 0
        dropped = 0
        for group_index in range(parquet.num_row_groups):
            table = parquet.read_row_group(group_index, columns=["conversations", "source"])
            sources = table.column("source").to_pylist()
            for offset, conversation in enumerate(table.column("conversations").to_pylist()):
                messages, unmapped = role_mapped(conversation)
                if unmapped:
                    for message in conversation:
                        if ROLE_MAPPING.get(message["from"]) is None:
                            unmapped_roles[message["from"]] += 1
                if trainable(messages):
                    groups[sources[offset]].append((file_index, row_base + offset))
                else:
                    dropped += 1
            row_base += table.num_rows
        scanned += row_base
        print(
            f"  scanned {path.name}: {row_base} rows, {dropped} untrainable", flush=True
        )
    return dict(groups), sum(len(v) for v in groups.values()), scanned, dict(unmapped_roles)


def quotas(groups: dict[str, list], total_rows: int, want: int) -> dict[str, int]:
    """Largest-remainder apportionment so the quotas sum to ``want`` exactly."""
    exact = {s: len(rows) * want / total_rows for s, rows in groups.items()}
    floors = {s: int(v) for s, v in exact.items()}
    short = want - sum(floors.values())
    for source, _ in sorted(exact.items(), key=lambda kv: -(kv[1] - floors[kv[0]]))[:short]:
        floors[source] += 1
    return floors


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-dir", required=True)
    ap.add_argument("--total", type=int, default=200_000)
    ap.add_argument("--output", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    files = sorted(Path(args.source_dir).glob("train-*.parquet"))
    if not files:
        raise SystemExit(f"no train-*.parquet under {args.source_dir}")
    print(f"sources: {len(files)} parquet shards")

    groups, total_rows, scanned, unmapped_roles = population(files)
    print(
        f"population: {total_rows} trainable of {scanned} scanned rows "
        f"({100 * total_rows / scanned:.2f}%) across {len(groups)} sources"
    )
    if unmapped_roles:
        print(f"unmapped roles skipped: {unmapped_roles}")
    print()

    quota = quotas(groups, total_rows, args.total)

    # Exact draw per source: the population already excludes untrainable rows, so no oversampling
    # and no post-hoc truncation is needed and every quota is filled by construction.
    wanted: dict[int, dict[int, str]] = collections.defaultdict(dict)
    for source, rows in sorted(groups.items()):
        for file_index, row_index in random.Random(args.seed).sample(rows, quota[source]):
            wanted[file_index][row_index] = source

    kept: dict[str, int] = collections.Counter()
    turn_hist: dict[str, int] = collections.Counter()
    chars_total = 0
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w") as sink:
        for file_index, path in enumerate(files):
            targets = wanted.get(file_index)
            if not targets:
                continue
            parquet = pq.ParquetFile(path)
            row_base = 0
            for group_index in range(parquet.num_row_groups):
                table = parquet.read_row_group(group_index, columns=["conversations"])
                rows = table.column("conversations").to_pylist()
                for offset, conversation in enumerate(rows):
                    row_index = row_base + offset
                    source = targets.get(row_index)
                    if source is None:
                        continue
                    messages, _ = role_mapped(conversation)
                    sink.write(
                        json.dumps(
                            {"id": f"pb-{file_index}-{row_index}", "conversations": messages},
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    kept[source] += 1
                    turn_hist[str(min(len(messages), 10))] += 1
                    chars_total += sum(len(m["content"]) for m in messages)
                row_base += table.num_rows
            print(f"  wrote from {path.name}: running total {sum(kept.values())}", flush=True)

    written = sum(kept.values())
    print(f"\nwrote {written} rows to {out_path}")
    print(f"\n{'source':44s} {'pop%':>7s} {'quota':>7s} {'kept':>7s} {'kept%':>7s} {'dev':>7s}")
    worst = 0.0
    for source in sorted(groups):
        pop_pct = 100 * len(groups[source]) / total_rows
        kept_pct = 100 * kept[source] / written if written else 0.0
        worst = max(worst, abs(kept_pct - pop_pct))
        print(
            f"{source:44s} {pop_pct:7.3f} {quota[source]:7d} {kept[source]:7d} "
            f"{kept_pct:7.3f} {kept_pct - pop_pct:+7.3f}"
        )
    print(f"\nworst per-source deviation from population: {worst:.3f} pp")
    print(f"mean chars/row: {chars_total / written:.0f} (est tokens {chars_total / written / 3.6:.0f})")

    # Gate G1: exact count and a population-faithful mix, else the run is not what was asked for.
    failures = []
    if written != args.total:
        failures.append(f"row count {written} != requested {args.total}")
    if worst >= 0.5:
        failures.append(f"worst per-source deviation {worst:.3f} pp >= 0.5 pp")
    if failures:
        print("\nG1 FAIL: " + "; ".join(failures))
        return 1
    print("\nG1 PASS: exact count, population-faithful per-source mix")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
