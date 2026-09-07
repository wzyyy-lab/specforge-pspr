"""Generate token-preserving HF teacher-prefix features for a declared holdout.

Uses the existing capture helper and original decode prompt renderer. These
are separate HF autoregressive trajectories, NOT exact decode_lattice replay.
Output partitions are fresh, never overwritten; skipped prompts are recorded.
This script does not train on calibration or validation prompts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("calibration", "validation"), required=True)
    parser.add_argument("--part", type=int, default=0)
    parser.add_argument("--parts", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-prompt-tokens", type=int, default=2816)
    parser.add_argument("--target-model", default="/kl_infra_infer_intern/wangzhuoyu/TAPS-SP/models/Qwen3-4B")
    parser.add_argument("--draft-config", default="configs/qwen3-4b-pspr-slotdeep.json")
    args = parser.parse_args()
    if args.parts < 1 or not 0 <= args.part < args.parts:
        parser.error("require 0 <= part < parts")
    if args.max_new_tokens < 2 or args.max_prompt_tokens < 1:
        parser.error("positive prompt budget and at least two new tokens required")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite holdout features: {args.output}")
    raw = args.prompts.read_bytes()
    prompts = [json.loads(line) for line in raw.decode().splitlines()]
    cfg = json.loads(Path(args.draft_config).read_text())
    layer_ids = cfg["dflash_config"]["target_layer_ids"]
    width = len(layer_ids) * cfg["hidden_size"]
    # Imports are deliberately after argument/output validation.
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from scripts.dump_eval_traces import capture
    from scripts.decode_lattice import build_input

    torch.manual_seed(20260905)
    device = torch.device("cuda")
    tok = AutoTokenizer.from_pretrained(args.target_model)
    target = AutoModelForCausalLM.from_pretrained(
        args.target_model, attn_implementation="sdpa", dtype=torch.bfloat16
    ).to(device).eval()
    args.output.mkdir(parents=True, exist_ok=False)
    manifest, skipped = [], []
    for index, row in enumerate(prompts):
        if index % args.parts != args.part:
            continue
        prompt = row["turns"][0]
        input_ids = build_input(tok, prompt, device)
        n_prompt = input_ids.shape[-1]
        if n_prompt > args.max_prompt_tokens:
            skipped.append(dict(index=index, id=row["id"], reason="prompt_token_limit", n_prompt=n_prompt))
            continue
        with torch.inference_mode():
            generated = target.generate(input_ids, do_sample=False, num_beams=1,
                max_new_tokens=args.max_new_tokens, eos_token_id=tok.eos_token_id,
                pad_token_id=tok.eos_token_id)
            full = generated[0].cpu().long()
            n_gen = full.numel() - n_prompt
            if n_gen < 2:
                skipped.append(dict(index=index, id=row["id"], reason="short_generation", n_gen=n_gen))
                continue
            mask = torch.zeros_like(full)
            mask[n_prompt:] = 1
            features = capture(target, full, layer_ids, device)
        assert features.shape == (full.numel(), width)
        assert torch.isfinite(features).all()
        name = f"pb_{args.split}_{index:04d}.ckpt"
        torch.save(dict(input_ids=full, loss_mask=mask, hidden_states=features), args.output / name)
        manifest.append(dict(dataset="perfectblend", split=args.split, source=row["source"],
            prompt_index=index, id=row["id"], file=name,
            prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
            token_ids_sha256=hashlib.sha256(full.numpy().tobytes()).hexdigest(),
            n_prompt=n_prompt, n_gen=n_gen))
        del features, full, generated, input_ids
        if len(manifest) % 16 == 0:
            print("HOLDOUT_CAPTURE", args.split, args.part, len(manifest), flush=True)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    provenance = dict(prompts=str(args.prompts.resolve()), prompts_sha256=hashlib.sha256(raw).hexdigest(),
        arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        teacher="HF target.generate, greedy, token IDs preserved without text roundtrip",
        scope="Fixed teacher-prefix proxy, not exact speculative decode replay", layer_ids=layer_ids,
        kept=len(manifest), skipped=skipped, generation_tokens=sum(m["n_gen"] for m in manifest))
    (args.output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print("HOLDOUT_CAPTURE_DONE", args.split, args.part, len(manifest), "skipped", len(skipped), flush=True)


if __name__ == "__main__":
    main()
