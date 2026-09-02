#!/usr/bin/env python3
"""Measure the label mismatch between SpecForge-style training and speculative decoding.

Two different notions of "correct next token" are in play, and if they diverge the draft is trained
for one task and evaluated on another:

* **Training (SpecForge / DFlash offline):** ``target_ids`` come from shifting ``input_ids``, so the
  label is *the next token of the corpus text*.
* **Decoding (acceptance):** a drafted token is accepted iff it equals the target model's own greedy
  argmax at that position (``block_ids == posterior``), so the label is *what the target would say*.

On a corpus of human-written assistant turns those two labels agree only where the target happens to
reproduce the human. This script measures that agreement rate directly, per dataset, restricted to
supervised (assistant) positions -- exactly the positions the objective trains on.

An agreement rate well below 1.0 means a fraction of the training signal actively teaches the draft
to predict tokens the target would never emit, which at decode time is indistinguishable from noise.

Usage:
    PYTHONPATH=. python scripts/measure_label_mismatch.py \
        --target-model /kl_infra_infer_intern/wangzhuoyu/TAPS-SP/models/Qwen3-4B \
        --data cache/dataset/sharegpt_train.jsonl --num-samples 40
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

SPECFORGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SPECFORGE))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-model", required=True)
    ap.add_argument("--data", required=True, help="prepared *_train.jsonl")
    ap.add_argument("--num-samples", type=int, default=40)
    ap.add_argument("--max-length", type=int, default=3072)
    ap.add_argument("--chat-template", default="qwen")
    ap.add_argument("--label", default=None, help="name for the report line")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    from specforge.data.preprocessing import preprocess_conversations
    from specforge.data.template import TEMPLATE_REGISTRY

    def to_sharegpt(row: dict) -> list[dict]:
        turns = row.get("conversations") or row.get("messages") or []
        out = []
        for t in turns:
            role = t.get("role")
            if role is None:
                role = "user" if t.get("from", "") in ("human", "user") else "assistant"
            out.append({"role": role, "content": t.get("value") or t.get("content") or ""})
        return out

    tokenizer = AutoTokenizer.from_pretrained(args.target_model)
    template = TEMPLATE_REGISTRY.get(args.chat_template)
    model = (
        AutoModelForCausalLM.from_pretrained(
            args.target_model, attn_implementation="sdpa", dtype=torch.bfloat16
        )
        .to("cuda")
        .eval()
    )

    rows = []
    with open(args.data) as handle:
        for line in handle:
            rows.append(json.loads(line))
            if len(rows) >= args.num_samples:
                break

    supervised = agree = 0
    per_sample = []
    for row in rows:
        conv = to_sharegpt(row)
        if not conv:
            continue
        res = preprocess_conversations(
            tokenizer, [conv], template, max_length=args.max_length
        )
        if not res["input_ids"]:
            continue
        input_ids = res["input_ids"][0][0].to(torch.long).unsqueeze(0).to("cuda")
        loss_mask = res["loss_mask"][0][0].to(torch.long).to("cuda")
        if input_ids.shape[1] < 2:
            continue
        with torch.no_grad():
            logits = model(input_ids).logits[0]
        # Position i predicts token i+1; the corpus label is input_ids[i+1] and the target's own label
        # is argmax(logits[i]). Compare them wherever the objective would supervise position i+1.
        greedy = logits[:-1].argmax(dim=-1)
        corpus = input_ids[0, 1:]
        mask = loss_mask[1:] > 0.5
        n = int(mask.sum())
        if n == 0:
            continue
        a = int((greedy[mask] == corpus[mask]).sum())
        supervised += n
        agree += a
        per_sample.append(a / n)

    name = args.label or Path(args.data).stem
    print(f"\n=== {name} ({len(per_sample)} samples, {supervised} supervised positions) ===")
    print(f"  P(target greedy == corpus next token) = {agree / max(supervised, 1):.4f}")
    if per_sample:
        per_sample.sort()
        print(f"  per-sample: min {per_sample[0]:.3f} | median "
              f"{per_sample[len(per_sample) // 2]:.3f} | max {per_sample[-1]:.3f}")
    print(
        "\n  Interpretation: 1 - this rate is the share of the training signal that teaches the\n"
        "  draft to emit tokens the target itself would not emit, and which therefore can never be\n"
        "  accepted at decode time."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
