#!/usr/bin/env python3
"""Attribution between the cascade arm and its blind control, from the training logs alone.

Why differencing the AGGREGATE selector loss is a valid attribution here, and would not be in general.

`selector_loss` under `profitable_repair` is `binary_gate_CE + conditional_alternative_CE`
(dflash_family_model.py:1041), and codex's D6 objection is that a change in the aggregate does not
prove anything about the gate.  That objection is defused by this specific pair of runs:

  * `selector_train_scope: err_only` freezes every ranker tensor, so `scores` are a fixed function of
    the data in BOTH arms and the alternative-CE term is identical sample for sample;
  * the two arms share the draft config, the objective, the anchor, `err_base_gain`, the LR schedule,
    `seed: 42`, the world size and the data order, and differ only by `selector_err_score_enabled`;
  * therefore the DIFFERENCE of the two aggregates is exactly the difference of the gate BCE terms,
    which is codex's `eval/cascade_gain_bce`.

The absolute values are still aggregates and are not interpreted as gate BCE.  Only the difference is.
"""

from __future__ import annotations

import ast
import pathlib
import re
import sys


def series(path: str) -> dict[int, dict]:
    out: dict[int, dict] = {}
    order = 0
    for line in pathlib.Path(path).read_text(errors="ignore").splitlines():
        i = line.find("{'train/")
        if i < 0:
            continue
        try:
            record = ast.literal_eval(line[i:])
        except Exception:
            continue
        if not isinstance(record, dict):
            continue
        # The step lives in the line PREFIX ("step 470: {...}"), not in the payload.  Parsing it is
        # not cosmetic: all 7 trainer ranks emit the same record, so positional recovery inflates the
        # step by 7x and would mislabel every row.
        match = re.search(r"step (\d+):\s*$", line[:i])
        if match is None:
            order += 1
            continue
        out[int(match.group(1))] = record
    return out


def main() -> None:
    cascade = series(sys.argv[1] if len(sys.argv) > 1 else "outputs/TRAIN_cascade.log")
    blind = series(sys.argv[2] if len(sys.argv) > 2 else "outputs/TRAIN_cascade_blind.log")
    common = sorted(set(cascade) & set(blind))
    print(f"matched points: {len(common)}   cascade={len(cascade)}  blind={len(blind)}")
    if not common:
        raise SystemExit("no matched log points")

    key = "train/selector_loss"
    print(
        f"\n{'step':>5} {'cascade_ce':>11} {'blind_ce':>11} {'gain_bce':>10} "
        f"{'cas_acc':>9} {'bli_acc':>9} {'d_acc':>9}"
    )
    for step in common:
        c, b = cascade[step], blind[step]
        ca = c.get("train/acc", float("nan"))
        ba = b.get("train/acc", float("nan"))
        print(
            f"{step:>5} {c[key]:>11.5f} {b[key]:>11.5f} {b[key] - c[key]:>+10.5f} "
            f"{ca:>9.5f} {ba:>9.5f} {ca - ba:>+9.5f}"
        )

    def window(steps):
        mc = sum(cascade[s][key] for s in steps) / len(steps)
        mb = sum(blind[s][key] for s in steps) / len(steps)
        return mc, mb, mb - mc

    for label, steps in (
        ("all", common),
        ("first 20%", common[: max(1, len(common) // 5)]),
        ("last 50%", common[len(common) // 2 :]),
        ("last 20%", common[-max(1, len(common) // 5) :]),
    ):
        mc, mb, gain = window(steps)
        print(f"{label:>10}: cascade={mc:.5f}  blind={mb:.5f}  gain_bce={gain:+.5f}")

    first = common[0]
    print(
        f"\nsanity, first logged point (step {first}): cascade={cascade[first][key]:.5f} vs "
        f"blind={blind[first][key]:.5f}; both arms start from the same anchor policy, so these must "
        f"be close -- |d| = {abs(cascade[first][key] - blind[first][key]):.5f}"
    )


if __name__ == "__main__":
    main()
