#!/usr/bin/env python3
"""HF target-greedy teacher-prefix features in the training-time tensor format.

WHY NOT JUST REUSE THE TRAINING CORPUS
--------------------------------------
Comparing a teacher-forced ladder measured on perfectblend against a decode ladder measured on six
benchmarks confounds two variables at once: block alignment (the thing under test) and corpus.  Worse,
the 7115-step run consumed one epoch of all 199293 perfectblend rows, so every such row is in-sample.

This is a SEPARATE HF-generated teacher-prefix proxy, not a tokenwise replay of decode_lattice.
The same prompt and target weights do not establish numerical trajectory identity between one-token
HF generation and block verification. The 2026-09-05 audit found 9/120 prompt lengths inconsistent
with decode even after allowing its untrimmed final block. The cause is not established here.
Comparing heads on identical files and anchors is valid for this fixed proxy population, but any
claim that ONLY anchor placement differs from real decoding requires persisted decode token IDs and
a tokenwise equality check, which this script does not implement.

NO RE-TOKENISATION
------------------
The hidden states are captured directly from ``[prompt_ids, generated_ids]``.  Round-tripping the
generation through text and back through ``preprocess_conversations`` would risk token-boundary drift
against ``decode_lattice.py:862``'s ``apply_chat_template`` rendering, which would silently move the
whole slot population.  ``loss_mask`` is 0 on the prompt and 1 on the generation, matching SpecForge's
assistant-span convention, so anchor sampling lands only inside the generated span exactly as it does
on corpus data.

Prompt selection is transcribed from ``decode_lattice.py:1174-1187`` for ``--eval-reserved`` with
``--max-samples 20 --shuffle-seed 2026``, i.e. the same 120 prompts as every DETAIL_*/REPLAY_* run.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SPECFORGE = Path(__file__).resolve().parents[1]
TAPS = Path("/home/wangzhuoyu/kl_infra_infer_intern/wangzhuoyu/TAPS-SP")
sys.path.insert(0, str(SPECFORGE))
sys.path.insert(0, str(TAPS))

from model import load_and_process_dataset, select_dataset_samples  # noqa: E402

RESERVED_20 = ("humaneval", "mt-bench")


@torch.inference_mode()
def capture(target, input_ids, layer_ids, device, offset=1):
    out = target(input_ids.unsqueeze(0).to(device), output_hidden_states=True, use_cache=False)
    sel = [out.hidden_states[L + offset][0] for L in layer_ids]
    feat = torch.cat(sel, dim=-1).to(torch.bfloat16).cpu()
    del out, sel
    return feat


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-model",
                    default="/kl_infra_infer_intern/wangzhuoyu/TAPS-SP/models/Qwen3-4B")
    ap.add_argument("--draft-config", default=str(SPECFORGE / "configs/qwen3-4b-pspr-cloze.json"))
    ap.add_argument("--output", required=True)
    ap.add_argument("--datasets", default="gsm8k,math500,humaneval,mbpp,alpaca,mt-bench")
    ap.add_argument("--max-samples", type=int, default=20)
    ap.add_argument("--shuffle-seed", type=int, default=2026)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--manifest", default=None)
    args = ap.parse_args()

    device = torch.device("cuda")
    cfg = json.load(open(args.draft_config))
    layer_ids = cfg["dflash_config"]["target_layer_ids"]
    width = len(layer_ids) * cfg["hidden_size"]

    tok = AutoTokenizer.from_pretrained(args.target_model)
    target = AutoModelForCausalLM.from_pretrained(
        args.target_model, attn_implementation="sdpa", dtype=torch.bfloat16
    ).to(device).eval()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    kept = skipped = 0

    for ds in args.datasets.split(","):
        ds = ds.strip()
        dataset = load_and_process_dataset(ds)
        reserved = 20 if ds in RESERVED_20 else 40
        dataset = select_dataset_samples(
            dataset, max_samples=reserved, sample_offset=0, shuffle_seed=args.shuffle_seed
        )
        idxs = list(range(min(len(dataset), args.max_samples)))
        for idx in idxs:
            uc = dataset[idx]["turns"][0]
            text = tok.apply_chat_template(
                [{"role": "user", "content": uc}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            prompt_ids = tok.encode(text, return_tensors="pt").to(device)
            n_prompt = prompt_ids.shape[1]
            with torch.inference_mode():
                gen = target.generate(
                    prompt_ids,
                    do_sample=False,
                    num_beams=1,
                    max_new_tokens=args.max_new_tokens,
                    eos_token_id=tok.eos_token_id,
                    pad_token_id=tok.eos_token_id,
                )
            full = gen[0].cpu().to(torch.long)
            n_gen = full.shape[0] - n_prompt
            if n_gen < 2:
                skipped += 1
                continue
            loss_mask = torch.zeros_like(full)
            loss_mask[n_prompt:] = 1
            feat = capture(target, full, layer_ids, device)
            assert feat.shape[0] == full.shape[0], f"{feat.shape} vs {full.shape}"
            assert feat.shape[1] == width, f"width {feat.shape[1]} != {width}"
            sha = hashlib.sha256(uc.encode()).hexdigest()
            name = f"{ds}_{idx:03d}.ckpt"
            torch.save(
                {"input_ids": full, "loss_mask": loss_mask, "hidden_states": feat},
                out_dir / name,
            )
            manifest.append(
                {"dataset": ds, "prompt_index": idx, "prompt_sha256": sha,
                 "file": name, "n_prompt": n_prompt, "n_gen": int(n_gen)}
            )
            kept += 1
            del feat, gen, full
        gc.collect()
        torch.cuda.empty_cache()
        print(f"[{ds}] kept={kept} skipped={skipped}", flush=True)

    dest = Path(args.manifest) if args.manifest else out_dir / "manifest.json"
    dest.write_text(json.dumps(manifest, indent=2))
    total_mb = sum(f.stat().st_size for f in out_dir.glob("*.ckpt")) / 2**20
    gen_tokens = sum(m["n_gen"] for m in manifest)
    print(f"done: kept={kept} skipped={skipped} gen_tokens={gen_tokens} "
          f"size={total_mb:.0f}MB -> {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
