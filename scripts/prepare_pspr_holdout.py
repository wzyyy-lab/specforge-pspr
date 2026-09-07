"""Prepare disjoint prompt-only PerfectBlend calibration/validation artifacts.

This does NOT regenerate targets, run a model, or edit any training dataset.
Exclude declared historical row IDs and every normalized user turn in them;
deduplicate first-user prompts within the remaining pool. This is exact
normalized-text exclusion, NOT semantic decontamination or a pretraining audit.
The old six-domain development benchmark is not used for threshold selection.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import unicodedata

import pyarrow.parquet as pq

from scripts.sample_perfectblend import quotas, role_mapped, trainable


def sha_file(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prompt_key(text):
    normalized = " ".join(unicodedata.normalize("NFKC", text).casefold().split())
    return hashlib.sha256(normalized.encode()).hexdigest()


def user_turns(row):
    messages = row.get("conversations", row.get("messages", []))
    for message in messages:
        role = message.get("role", message.get("from"))
        if role in {"human", "user"}:
            text = message.get("content", message.get("value"))
            if not isinstance(text, str):
                raise ValueError("non-text user turn in exclusion input")
            if text.strip():
                yield text


def exclusions(paths):
    ids, prompts, manifest = set(), set(), []
    for path in paths:
        count, turns = 0, 0
        with path.open() as stream:
            for line in stream:
                row = json.loads(line)
                count += 1
                if row.get("id") is not None:
                    ids.add(str(row["id"]))
                for text in user_turns(row):
                    prompts.add(prompt_key(text))
                    turns += 1
        manifest.append(dict(path=str(path.resolve()), sha256=sha_file(path),
                             rows=count, user_turns=turns))
    return ids, prompts, manifest


def rows(files):
    for file_index, path in enumerate(files):
        row_index = 0
        for batch in pq.ParquetFile(path).iter_batches(
            batch_size=1024, columns=["conversations", "source"]
        ):
            for row in batch.to_pylist():
                yield f"pb-{file_index}-{row_index}", row
                row_index += 1


def prepare(source_dir, excluded_paths, out, count, seed, max_chars):
    if out.exists():
        raise FileExistsError(f"refusing to overwrite holdout artifacts: {out}")
    files = sorted(source_dir.glob("train-*.parquet"))
    if not files or not excluded_paths or count < 1 or max_chars < 1:
        raise ValueError("need parquet sources, explicit exclusions and positive sizes")
    excluded_ids, excluded_prompts, excluded_manifest = exclusions(excluded_paths)
    print("EXCLUSIONS", len(excluded_ids), len(excluded_prompts), flush=True)
    seen = set(excluded_prompts)
    groups, dropped = defaultdict(list), Counter()
    for row_id, row in rows(files):
        if row_id in excluded_ids:
            dropped["declared_row_id"] += 1
            continue
        messages, _ = role_mapped(row["conversations"])
        prompt = next((m["content"] for m in messages if m["role"] == "user"), None)
        if not trainable(messages) or not isinstance(prompt, str) or not prompt.strip():
            dropped["missing_prompt_or_assistant"] += 1
            continue
        if len(prompt) > max_chars:
            dropped["prompt_char_limit"] += 1
            continue
        key = prompt_key(prompt)
        if key in seen:
            dropped["declared_or_duplicate_prompt"] += 1
            continue
        seen.add(key)
        groups[row["source"]].append((row_id, key))
    population = sum(map(len, groups.values()))
    if population < 2 * count:
        raise ValueError(f"only {population} unique eligible prompts for {2*count} requested")
    pool_quota = quotas(groups, population, 2 * count)
    rng = random.Random(seed)
    draws = {s: rng.sample(groups[s], pool_quota[s]) for s in sorted(groups)}
    calibration_quota = quotas(draws, 2 * count, count)
    wanted = {}
    for source, selected in draws.items():
        for i, (row_id, key) in enumerate(selected):
            split = "calibration" if i < calibration_quota[source] else "validation"
            wanted[row_id] = dict(split=split, source=source, normalized_prompt_sha256=key)
    selected_rows = {"calibration": [], "validation": []}
    for row_id, row in rows(files):
        info = wanted.get(row_id)
        if info is None:
            continue
        messages, _ = role_mapped(row["conversations"])
        prompt = next(m["content"] for m in messages if m["role"] == "user")
        assert prompt_key(prompt) == info["normalized_prompt_sha256"]
        assert row_id not in excluded_ids and prompt_key(prompt) not in excluded_prompts
        selected_rows[info["split"]].append(dict(
            id=row_id, source=info["source"], turns=[prompt],
            conversations=[dict(role="user", content=prompt)]))
    for records in selected_rows.values():
        assert len(records) == count
        rng.shuffle(records)
    assert len({prompt_key(r["turns"][0]) for records in selected_rows.values()
                for r in records}) == 2 * count
    out.mkdir(parents=True, exist_ok=False)
    outputs = {}
    for split, records in selected_rows.items():
        path = out / f"{split}.jsonl"
        with path.open("x") as sink:
            for record in records:
                sink.write(json.dumps(record, ensure_ascii=False) + "\n")
        outputs[split] = dict(path=str(path.resolve()), sha256=sha_file(path),
                              count=len(records), source_counts=dict(Counter(r["source"] for r in records)))
    manifest = dict(
        seed=seed, per_split=count, max_prompt_chars=max_chars,
        normalization="NFKC, casefold, collapse Unicode whitespace, SHA256",
        scope="Head-training declared-file row-ID/all-user-turn exclusion and within-pool first-prompt dedup only; not semantic/pretraining decontamination",
        target_labels="none; prompt-only, not yet regenerated or captured",
        sources=[dict(path=str(p.resolve()), bytes=p.stat().st_size, sha256=sha_file(p)) for p in files],
        exclusions=excluded_manifest, eligible_unique=population,
        population_by_source={k: len(v) for k, v in groups.items()}, dropped=dict(dropped),
        outputs=outputs, selections=wanted,
    )
    with (out / "manifest.json").open("x") as sink:
        json.dump(manifest, sink, ensure_ascii=False, indent=2)
        sink.write("\n")
    print("PSPR_HOLDOUT_PREPARED", json.dumps(outputs), flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--exclude-jsonl", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--per-split", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--max-prompt-chars", type=int, default=8000)
    args = parser.parse_args()
    prepare(args.source_dir, args.exclude_jsonl, args.out,
            args.per_split, args.seed, args.max_prompt_chars)


if __name__ == "__main__":
    main()
