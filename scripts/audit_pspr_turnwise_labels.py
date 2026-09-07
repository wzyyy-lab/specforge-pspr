"""Quality-control sample of the NEW turnwise tokenized training artifact.

HF teacher-forced target/corpus agreement, not speculative acceptance or a
paired comparison to the old cache population. No parameters are trained.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np

from scripts.prepare_pspr_regen_turns import sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=128)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest_path = args.data.with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text())
    assert sha(args.data) == manifest["output_sha256"]
    n = manifest["counts"]["kept_turns"]
    assert 1 <= args.samples <= n
    indices = set(np.random.default_rng(20260905).choice(n, args.samples, replace=False).tolist())
    rows = []
    with args.data.open() as stream:
        for i, line in enumerate(stream):
            if i in indices:
                rows.append((i, json.loads(line)))
    assert len(rows) == args.samples
    import torch
    from transformers import AutoModelForCausalLM
    target_path = Path(manifest["arguments"]["target_model"])
    target = AutoModelForCausalLM.from_pretrained(target_path, dtype=torch.bfloat16,
        attn_implementation="sdpa").to("cuda").eval()
    per_sample = []
    for index, row in rows:
        ids = torch.tensor(row["input_ids"], device="cuda")[None]
        mask = np.asarray(row["loss_mask"])
        positions = np.flatnonzero(mask)
        assert positions.tolist() == list(range(row["prefix_tokens"], ids.shape[1]))
        assert len(positions) == row["supervised_tokens"] >= 2
        prediction_positions = torch.tensor(positions-1, device="cuda")
        truth = ids[0,prediction_positions+1]
        with torch.inference_mode():
            logits = target(ids, attention_mask=torch.ones_like(ids), use_cache=False,
                            logits_to_keep=prediction_positions).logits[0]
            assert torch.isfinite(logits).all()
            values, predicted = logits.max(-1)
            label_values = logits.gather(-1,truth[:,None]).squeeze(-1)
            wrong = predicted.ne(truth)
            result = dict(row=index,id=row["id"],source_id=row["source_id"],
                tokens=truth.numel(),mismatches=int(wrong.sum()),
                mismatches_gap_ge_1=int((wrong & ((values-label_values)>=1)).sum()),
                input_ids_sha256=hashlib.sha256(np.asarray(row["input_ids"],dtype=np.int64).tobytes()).hexdigest())
        per_sample.append(result)
        del ids, prediction_positions, truth, logits, values, predicted, label_values, wrong
        if len(per_sample)%16 == 0:
            print("TURN_LABEL_QC",len(per_sample),flush=True)
    totals = {k:sum(r[k] for r in per_sample) for k in ("tokens","mismatches","mismatches_gap_ge_1")}
    paths = [Path(__file__).resolve(), Path("scripts/prepare_pspr_regen_turns.py").resolve(),
             manifest_path.resolve(), *sorted(target_path.glob("*.json")), *sorted(target_path.glob("*.safetensors"))]
    output = dict(created_utc=datetime.now(timezone.utc).isoformat(),scope=__doc__,
        data=str(args.data),data_sha256=manifest["output_sha256"],
        source_sha256={str(p):sha(p) for p in paths},seed=20260905,samples=args.samples,
        totals=totals,mismatch_rate=totals["mismatches"]/totals["tokens"],
        package_versions={k:importlib.metadata.version(k) for k in ("torch","transformers")},
        per_sample=per_sample)
    with args.output.open("x") as sink:
        json.dump(output,sink,indent=2)
        sink.write("\n")
    print("TURN_LABEL_QC_DONE",json.dumps(totals),flush=True)


if __name__ == "__main__":
    main()
