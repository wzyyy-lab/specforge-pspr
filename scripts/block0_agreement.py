#!/usr/bin/env python3
"""Block-0 decision agreement between two decode runs -- the only zero-noise comparison available.

Greedy speculative decoding is a closed loop: the moment two runs pick a different token, every later
block sees a different prefix, so per-block accept lengths stop being comparable.  Block 0 is the
exception.  Its prefix is the prompt itself, identical by construction in both runs, so
``acceptance_lengths[0]`` is a comparison of the DECISION RULES with no trajectory confound.

Two rules that are algebraically identical must therefore agree on block 0 for 100% of prompts, up to
whatever run-to-run nondeterminism the decoder itself has.  Quantifying that floor is the whole point
of passing a self-replay as one of the pairs: without it, "107/120" cannot be read either way.

Also reports how deep into the generation the two runs stay identical.  That number is NOT a measure of
rule agreement -- it decays geometrically once any single decision differs -- but it separates "the runs
diverge immediately" from "the runs agree for a long time and then drift".
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib


def load(path: str, mode: str | None) -> dict[tuple[str, str], list[int]]:
    records = json.load(open(path))["results"]
    modes = {r.get("mode") for r in records}
    if mode is None:
        if len(modes) != 1:
            raise SystemExit(f"{path} holds modes {sorted(modes)}; pass an explicit mode")
        mode = next(iter(modes))
    out = {}
    for r in records:
        if r.get("mode") != mode:
            continue
        out[(r["dataset"], r["prompt_sha256"])] = r["acceptance_lengths"]
    if not out:
        raise SystemExit(f"{path} has no records for mode={mode!r} (available: {sorted(modes)})")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--mode-a", default=None)
    ap.add_argument("--mode-b", default=None)
    ap.add_argument("--label", default=None)
    args = ap.parse_args()

    a = load(args.a, args.mode_a)
    b = load(args.b, args.mode_b)
    keys = sorted(set(a) & set(b))
    if not keys:
        raise SystemExit("no prompts in common")
    label = args.label or f"{pathlib.Path(args.b).stem} vs {pathlib.Path(args.a).stem}"

    agree = sum(1 for k in keys if a[k][0] == b[k][0])
    delta = collections.Counter(b[k][0] - a[k][0] for k in keys)
    prefix = []
    for k in keys:
        n = 0
        for x, y in zip(a[k], b[k]):
            if x != y:
                break
            n += 1
        prefix.append((n, len(a[k])))
    whole = sum(1 for n, total in prefix if n == total)
    lengths = sorted(n for n, _ in prefix)

    print(f"{label}   ({len(keys)} matched prompts)")
    print(f"  block-0 accept length identical : {agree}/{len(keys)} = {100 * agree / len(keys):.1f}%")
    print(f"  block-0 delta histogram        : {dict(sorted(delta.items()))}")
    print(f"  identical for the whole run    : {whole}/{len(keys)}")
    print(
        f"  identical block prefix         : median {lengths[len(lengths) // 2]}, "
        f"min {lengths[0]}, max {lengths[-1]}"
    )


if __name__ == "__main__":
    main()
