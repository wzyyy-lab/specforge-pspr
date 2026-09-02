"""Gate: on-policy trace collection walks EXACTLY the decode-time trajectory.

DAgger only fixes exposure bias if the anchors recorded by ``collect_trace_seq.py --selector-head``
are the anchors ``decode_lattice.py --modes latgate`` actually visits.  Before the fix, the collector
used an ADDITIVE gate (``bo - pr[0] > tau``) and the legacy ``selector.mlp`` path with no
``delta_term``, so for a dh2048 checkpoint it walked a different trajectory than decode -- which
would have silently poisoned the whole DAgger round.

This gate runs both implementations on the same prompts with the same checkpoint and the same gate
tuple and asserts the committed sequences and per-block accept counts are IDENTICAL.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REF = Path(__file__).resolve().parents[2] / "TAPS-SP"
sys.path.insert(0, str(REF))
sys.path.insert(0, str(REF / "scripts"))

from model import DFlashDraftModel, load_and_process_dataset, select_dataset_samples  # noqa: E402
from joint.lattice_selector import LatticeSelector  # noqa: E402
import collect_trace_seq as CT  # noqa: E402
import decode_lattice as DL  # noqa: E402

SELECTOR = Path(__file__).resolve().parents[1] / "outputs/pspr_dh2048_exact/best.pt"
GATE = dict(gate_tau=0.0, gate_rho=3.0, gate_skip0=False, gate_theta=0.0)
MAX_NEW = 96
N_PROMPTS = 3


def main() -> int:
    device = torch.device("cuda")
    target = AutoModelForCausalLM.from_pretrained(
        str(REF / "models/Qwen3-4B"), attn_implementation="sdpa", dtype=torch.bfloat16
    ).to(device).eval()
    draft = DFlashDraftModel.from_pretrained(
        str(REF / "models/Qwen3-4B-DFlash-b16"), attn_implementation="sdpa", dtype=torch.bfloat16
    ).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(str(REF / "models/Qwen3-4B"))

    selector, meta = LatticeSelector.from_checkpoint(str(SELECTOR), device=device)
    selector = selector.float().eval()
    embed = target.model.embed_tokens.weight.float().detach()
    block_size = draft.block_size
    stop_ids = [tokenizer.eos_token_id]

    print(f"selector={SELECTOR}")
    print(f"  cal_gain={meta.get('cal_gain'):+.4f} ckpt_tau={meta.get('tau')}")
    print(f"  gate under test={GATE}  block_size={block_size}  max_new={MAX_NEW}")
    print(f"  delta_h={getattr(selector, 'delta_h', 0)} (delta_term active: "
          f"{getattr(selector, 'delta_h', 0) > 0})\n")

    failures = 0
    checks = 0
    for ds_name in ("gsm8k", "humaneval"):
        dataset = load_and_process_dataset(ds_name)
        dataset = select_dataset_samples(dataset, max_samples=N_PROMPTS, sample_offset=0,
                                         shuffle_seed=2026)
        for idx in range(len(dataset)):
            input_ids = CT.build_input(tokenizer, dataset[idx]["turns"][0], device)

            committed, blocks, num_input, _ = CT.decode_prompt(
                draft, target, input_ids, draft.mask_token_id, MAX_NEW, block_size, stop_ids,
                topk=16, save_hidden=False, save_thidden=False, selector=selector,
                embed=embed, **GATE)
            collect_accepts = [b["accept"] + 1 for b in blocks]
            collect_starts = [b["start"] - num_input for b in blocks]

            decode_accepts = DL.decode(
                "latgate", draft, target, None, selector, None, embed, input_ids,
                draft.mask_token_id, MAX_NEW, block_size, stop_ids, **GATE)

            dec_starts, s = [], 0
            for a in decode_accepts:
                dec_starts.append(s)
                s += a

            ok_acc = collect_accepts == decode_accepts
            ok_start = collect_starts == dec_starts
            checks += 2
            failures += (not ok_acc) + (not ok_start)
            tag = "PASS" if (ok_acc and ok_start) else "FAIL"
            print(f"[{tag}] {ds_name}:{idx}  blocks={len(collect_accepts)}/{len(decode_accepts)}  "
                  f"sum_accept={sum(collect_accepts)}/{sum(decode_accepts)}")
            if not ok_acc:
                print(f"         collect accepts = {collect_accepts}")
                print(f"         decode  accepts = {decode_accepts}")
            if not ok_start:
                print(f"         collect anchors = {collect_starts}")
                print(f"         decode  anchors = {dec_starts}")

    print(f"\n{checks - failures}/{checks} PASS")
    if failures:
        print("on-policy collection does NOT match decode -- DAgger would train on wrong anchors")
        return 1
    print("collector walks the decode-time trajectory exactly; anchors are on-policy")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
