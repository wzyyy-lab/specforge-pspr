#!/usr/bin/env python3
"""Materialise the UNTRAINED cascade (champion warm start, zero updates) as an exportable checkpoint.

This is the control that decides what the negative result means.

At step 0 the cascade's gate logit is the parameter-free anchor `logsumexp(s_1:) - s0 - log(rho)`,
`err_base_gain` is 0 and the score read-out is 0, so the factorized MAP rule reduces algebraically to
`p_alt > rho * p0` -- the champion's deployed `latgate rho=3`.  `scripts/gate_pspr_cascade.py` proves
that on synthetic tensors and `scripts/smoke_pspr_cascade_e2e.py` proves the whole export/reload/decode
chain at toy scale, but neither proves it on the real 42M selector through the real six-domain decoder.

Why that matters right now: the trained cascade loses 0.2348 macro to the champion, and no
`--repair-margin` recovers it (accept falls monotonically for m >= 0.5).  There are exactly two
explanations and they call for opposite next moves:

  A) the anchor/serving path is subtly wrong, so the arm never actually started from the champion --
     then the 500-step result says nothing about the idea and the plumbing must be fixed;
  B) the anchor is exact and TRAINING moved the decision function somewhere worse for accept length --
     then the plumbing is fine and the objective is the problem.

Decoding this artefact at `--repair-margin 0` and paired-comparing against the champion's
`latgate rho=3` separates them: under (B) every per-prompt difference is exactly 0.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import torch

SPECFORGE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SPECFORGE))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--draft-config", required=True)
    ap.add_argument("--champion", required=True, help="champion training_state.pt or its run dir")
    ap.add_argument("--out", required=True, help="directory to write training_state.pt into")
    args = ap.parse_args()

    from transformers import Qwen3Config

    from specforge.modeling.draft.pspr_cascade import PSPRCascadeDraftModel
    from specforge.training.model_loading import warm_start_draft_model

    payload = json.loads(pathlib.Path(args.draft_config).read_text())
    config = Qwen3Config(**{k: v for k, v in payload.items() if k != "dflash_config"})
    config._attn_implementation = "eager"
    config.dflash_config = payload["dflash_config"]
    model = PSPRCascadeDraftModel(config)

    report = warm_start_draft_model(
        model,
        args.champion,
        draft_config=model.config,
        strategy="dflash",
    )
    print(
        f"warm start: loaded {report.loaded_keys} tensors from {report.checkpoint_format} "
        f"checkpoint; fresh = {sorted(report.missing_keys)}"
    )

    selector = model.candidate_selector
    readout = float(selector.err_score_head[-1].weight.abs().sum()) + float(
        selector.err_score_head[-1].bias.abs().sum()
    )
    gain = float(selector.err_base_gain["gain"].detach())
    if readout != 0.0 or gain != 0.0:
        raise SystemExit(
            f"the untrained cascade is not at its step-0 contract: |readout|={readout:g}, "
            f"base_gain={gain:g}. Exporting it would silently NOT be the champion-equivalent policy."
        )
    print(f"step-0 contract holds: |readout| = {readout:g}, base_gain = {gain:g}")

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    # The provenance fields the exporter and decode_lattice.validate_selector_policy require.  They
    # describe what this artefact IS -- a zero-update cascade under the profitable_repair contract --
    # not a run that happened.
    torch.save(
        {
            "draft_state_dict": model.state_dict(),
            "strategy": "dflash",
            "global_step": 0,
            "dflash2_selector_objective": "profitable_repair",
        },
        out / "training_state.pt",
    )
    print(f"wrote {out / 'training_state.pt'}")


if __name__ == "__main__":
    main()
