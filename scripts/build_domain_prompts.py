#!/usr/bin/env python3
"""Build a code+chat prompt pool for target-greedy regeneration, provably disjoint from the eval set.

WHY THESE SOURCES
-----------------
Direct standardization (`scripts/analyze_transfer_gap.py`) puts the selector's conditional deficit
entirely in code and chat, with maths at zero:

    gsm8k +0.03pp   math500 -0.82pp   mbpp -6.04pp   mt-bench -6.72pp   humaneval -8.11pp
    alpaca -8.63pp

The obvious response -- train on more of the eval datasets -- would be test-set adaptation dressed up
as a method improvement.  So the pool deliberately contains NO row of any eval dataset except mbpp's
non-test splits:

    CodeAlpaca-20k          code instructions, not an eval dataset
    livecodebench prompts   competitive programming, not an eval dataset
    mbpp train/validation   the eval uses `sanitized/test`, so these splits are disjoint
    ShareGPT                multi-domain chat, not an eval dataset

Consequently, any improvement on humaneval (-8.11pp), mt-bench (-6.72pp) or alpaca (-8.63pp) is
generalisation to unseen datasets, not adaptation.  That is the whole point of choosing them, and it
is what makes the experiment falsifiable rather than self-confirming.

The reserved-prompt collision check is belt and braces: the sources should not overlap by
construction, so a non-zero collision count means an assumption is wrong and the build aborts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import random
import sys
from pathlib import Path

TAPS = Path("/home/wangzhuoyu/kl_infra_infer_intern/wangzhuoyu/TAPS-SP")
sys.path.insert(0, str(TAPS))

EVAL_DATASETS = ("gsm8k", "math500", "humaneval", "mbpp", "alpaca", "mt-bench")
RESERVED = {"humaneval": 20, "mt-bench": 20}
RESERVED_DEFAULT = 40


def norm(text: str) -> str:
    return " ".join(text.split()).strip().lower()


def reserved_prompts(shuffle_seed: int) -> set[str]:
    """Exactly the prompts `decode_lattice.py --eval-reserved` evaluates on."""
    from model.utils import load_and_process_dataset, select_dataset_samples

    out = set()
    for ds in EVAL_DATASETS:
        data = load_and_process_dataset(ds)
        n = RESERVED.get(ds, RESERVED_DEFAULT)
        picked = select_dataset_samples(data, max_samples=n, sample_offset=0,
                                       shuffle_seed=shuffle_seed)
        for row in picked:
            for turn in row["turns"]:
                out.add(norm(turn))
    return out


def collect(name: str, limit: int | None):
    """Return [(id, prompt)] for one source."""
    from datasets import load_dataset
    from model.utils import _local_dataset_file, load_and_process_dataset

    rows = []
    if name == "codealpaca":
        data = load_and_process_dataset("codealpaca")
        rows = [(f"ca-{i}", r["turns"][0]) for i, r in enumerate(data)]
    elif name == "livecodebench":
        data = load_and_process_dataset("livecodebench")
        key = "turns" if "turns" in data.column_names else None
        for i, r in enumerate(data):
            t = r["turns"][0] if key else (r.get("question_content") or r.get("prompt") or "")
            if t:
                rows.append((f"lcb-{i}", t))
    elif name == "mbpp-train":
        # The eval reads `sanitized/test`; train and validation are disjoint splits of the same
        # dataset, which is the closest legitimate in-distribution code source available.
        # `_local_dataset_file` points at TAPS-SP/hf_assets, which does not exist in this checkout;
        # the parquet files live in the ordinary HF hub cache, so read them from there.
        snap = sorted(pathlib.Path(
            "~/.cache/huggingface/hub/datasets--google-research-datasets--mbpp/snapshots"
        ).expanduser().glob("*/sanitized"))
        for split in ("train", "validation", "prompt"):
            files = [str(d / f"{split}-00000-of-00001.parquet") for d in snap
                     if (d / f"{split}-00000-of-00001.parquet").exists()]
            if not files:
                continue
            d = load_dataset("parquet", data_files={split: files}, split=split)
            rows += [(f"mbppt-{split}-{i}", r["prompt"]) for i, r in enumerate(d)]
    elif name == "sharegpt":
        data = load_and_process_dataset("sharegpt")
        rows = [(f"sg-{i}", r["turns"][0]) for i, r in enumerate(data) if r["turns"]]
    else:
        raise ValueError(name)
    rows = [(i, t) for i, t in rows if t and t.strip()]
    if limit is not None and len(rows) > limit:
        random.Random(1234).shuffle(rows)
        rows = rows[:limit]
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--shuffle-seed", type=int, default=0,
                    help="must match decode_lattice.py's --shuffle-seed so the reserved set is the "
                         "same one the evaluation uses")
    ap.add_argument("--sources", default="codealpaca:20000,mbpp-train:600,"
                                         "livecodebench:1500,sharegpt:45000")
    ap.add_argument("--min-chars", type=int, default=16)
    ap.add_argument("--max-chars", type=int, default=6000)
    args = ap.parse_args()

    print("collecting the reserved eval prompts ...", flush=True)
    reserved = reserved_prompts(args.shuffle_seed)
    print(f"  {len(reserved)} reserved prompt strings over {len(EVAL_DATASETS)} datasets")

    seen, out, per_source, collisions = set(), [], {}, 0
    for spec in args.sources.split(","):
        name, _, cap = spec.partition(":")
        rows = collect(name.strip(), int(cap) if cap else None)
        kept = 0
        for rid, text in rows:
            if not (args.min_chars <= len(text) <= args.max_chars):
                continue
            key = norm(text)
            if key in reserved:
                collisions += 1
                continue
            h = hashlib.sha1(key.encode()).hexdigest()
            if h in seen:
                continue
            seen.add(h)
            out.append({"id": f"dom-{rid}", "conversations": [{"role": "user", "content": text}]})
            kept += 1
        per_source[name] = kept
        print(f"  {name:<14} {len(rows):>7} available -> {kept:>7} kept", flush=True)

    print(f"\ncollisions with the reserved eval prompts: {collisions}")
    if collisions:
        print("ABORT: a source overlaps the evaluation set; the disjointness argument is void")
        return 1

    random.Random(7).shuffle(out)
    dest = Path(args.output)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w") as f:
        for row in out:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {dest} ({len(out)} prompts)")
    print("composition: " + "  ".join(f"{k} {v}" for k, v in per_source.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
