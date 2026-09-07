"""Run the original speculative decoder on external prompt-only holdout JSONL.

No decoding logic is copied or changed. Native exported rho/tau/theta are
validated, and every prompt's accepted-prefix counts are saved for pairing.
This is target agreement, not benchmark answer quality or a speed benchmark.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-prompt-tokens", type=int, default=2816)
    parser.add_argument("--max-samples", type=int, help="explicit small-sample smoke only; default all prompts")
    parser.add_argument("--gate-stats", action="store_true", help="save reachable-policy counters per prompt")
    parser.add_argument("--calibration", type=Path, help="independently fitted immutable global-rho artifact")
    parser.add_argument("--calibration-arm", help="model label inside --calibration; both arguments required together")
    parser.add_argument("--target-model", default="/kl_infra_infer_intern/wangzhuoyu/TAPS-SP/models/Qwen3-4B")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite holdout evaluation: {args.output}")
    if args.max_new_tokens < 1 or args.max_prompt_tokens < 1:
        parser.error("token budgets must be positive")
    if args.max_samples is not None and args.max_samples < 1:
        parser.error("max samples must be positive")
    if bool(args.calibration) != bool(args.calibration_arm):
        parser.error("--calibration and --calibration-arm must be supplied together")
    # Existing helper puts both project roots on sys.path, then import the
    # exact function used by the six-domain CLI (not a decoder reimplementation).
    from scripts.dump_eval_traces import capture as _capture
    import scripts.decode_lattice as reference
    import torch

    rows = [json.loads(line) for line in args.prompts.read_text().splitlines()]
    if args.max_samples is not None:
        rows = rows[:args.max_samples]
    calibrated_policy, calibration_provenance = None, None
    if args.calibration:
        from scripts.calibrate_pspr_gate import validate_for_evaluation
        calibrated_policy, calibration_provenance = validate_for_evaluation(
            args.calibration, args.calibration_arm, args.export,
            args.target_model, args.prompts, reference.__file__)
    torch.manual_seed(20260905)
    device = torch.device("cuda")
    target = reference.AutoModelForCausalLM.from_pretrained(
        args.target_model, attn_implementation="sdpa", dtype=torch.bfloat16
    ).to(device).eval()
    draft = reference.DFlashDraftModel.from_pretrained(
        str(args.export / "backbone"), attn_implementation="sdpa", dtype=torch.bfloat16
    ).to(device).eval()
    tokenizer = reference.AutoTokenizer.from_pretrained(args.target_model)
    selector, payload = reference.load_selector_checkpoint(str(args.export / "selector.pt"), device=device)
    selector = selector.float().eval()
    exported_policy = payload["decode_policy"]
    policy = dict(exported_policy)
    gate = SimpleNamespace(allow_policy_mismatch=False, beam=1, path_branch=1, repair_margin=0.,
        gate_rho=policy["selector_gate_rho"], gate_tau=policy["selector_gate_tau"],
        gate_theta=policy["selector_gate_theta"], gate_skip0=policy["selector_gate_skip_first"])
    reference.validate_selector_policy(payload, gate, {"latgate"})
    if calibrated_policy is not None:
        if calibration_provenance["original_policy"] != exported_policy:
            raise ValueError("calibration exported policy mismatch")
        policy = calibrated_policy
        gate.gate_rho = policy["selector_gate_rho"]
        # Original export contract was validated above. This explicitly labelled
        # calibration changes only rho, never disables all policy validation.
        print("INDEPENDENT_RHO_CALIBRATION", json.dumps(calibration_provenance), flush=True)
    embed = target.model.embed_tokens.weight.float().detach()
    if reference.uses_slot_z(selector):
        selector.bind_target_embedding(embed)
    assert draft.block_size == 16
    results, skipped, gate_totals = [], [], Counter()
    for index, row in enumerate(rows):
        prompt = row["turns"][0]
        ids = reference.build_input(tokenizer, prompt, device)
        if ids.shape[-1] > args.max_prompt_tokens:
            skipped.append(dict(index=index, id=row["id"], reason="prompt_token_limit"))
            continue
        stats = defaultdict(int) if args.gate_stats else None
        accepted = reference.decode("latgate", draft, target, None, selector, None,
            embed, ids, draft.mask_token_id, args.max_new_tokens, draft.block_size,
            [tokenizer.eos_token_id], gate_rho=gate.gate_rho, gate_tau=gate.gate_tau,
            gate_theta=gate.gate_theta, gate_skip0=gate.gate_skip0, stats=stats)
        assert accepted and all(1 <= n <= 16 for n in accepted)
        results.append(dict(dataset="perfectblend", source=row["source"], id=row["id"],
            prompt_index=index, prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
            mode="latgate", acceptance_lengths=accepted, accepted_sum=sum(accepted),
            num_blocks=len(accepted), mean_acceptance=sum(accepted)/len(accepted)))
        if stats is not None:
            if any(k.startswith("_") for k in stats):
                raise ValueError("decoder left transient diagnostic state")
            assert stats["blocks"] == len(accepted)
            assert stats["accepted_slots"] + stats["blocks"] == sum(accepted)
            assert stats["base_right_kept"] + stats["base_right_destroyed"] == stats["base_right_n"]
            assert stats["fix_recovered"] + stats["fix_kept_wrong"] + stats["fix_wrong_override"] == stats["fixable_n"]
            results[-1]["gate_stats"] = dict(stats)
            gate_totals.update(stats)
        if len(results) % 16 == 0:
            print("HOLDOUT_DECODE", len(results), "of", len(rows), flush=True)
    output = dict(arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        block_size=16, proposal_slots=15, results=results, skipped=skipped,
        decode_policy=policy, prompts_sha256=sha(args.prompts), selector_sha256=sha(args.export / "selector.pt"),
        reference_decoder=str(Path(reference.__file__).resolve()),
        reference_decoder_sha256=sha(reference.__file__),
        scope="Original native speculative decoding on new prompt split; target agreement, not answer quality or speed")
    if args.gate_stats:
        output["gate_stats"] = dict(gate_totals)
    if calibration_provenance is not None:
        output["calibration_provenance"] = calibration_provenance
        output["exported_decode_policy"] = exported_policy
        output["scope"] = "Independently calibrated global rho with immutable weights; original speculative decoder; target agreement, not answer quality, throughput or risk guarantee"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as sink:
        json.dump(output, sink, indent=2)
        sink.write("\n")
    print("HOLDOUT_DECODE_DONE", len(results), "skipped", len(skipped), args.output, flush=True)


if __name__ == "__main__":
    main()
