#!/bin/bash
# Target-greedy regeneration of the domain (code+chat) prompt pool.
# Settings are byte-identical to the documented PerfectBlend regen
# (PSPR_Training_Route_A_vs_Route_B.md:178-190) so that the ONLY variable
# between the two corpora is the prompt distribution:
#   --temperature 0  --reasoning disable  --max-tokens 2048
set -e
cd /home/wangzhuoyu/kl_infra_infer_intern/wangzhuoyu/SpecForge

IN=cache/dataset/domain_codechat_prompts_clean.jsonl
OUT=cache/dataset/domain_codechat_qwen3-4b_regen.jsonl

# Drop every prompt drawn from a dataset that the 6-domain eval also uses
# (mbpp).  CodeAlpaca-20k and ShareGPT-52k are not eval datasets, so any
# gain on humaneval/mbpp/mt-bench/alpaca is cross-dataset generalisation.
python3 - <<'PYEOF'
import json
src = "cache/dataset/domain_codechat_prompts.jsonl"
dst = "cache/dataset/domain_codechat_prompts_clean.jsonl"
kept, dropped = 0, 0
with open(src) as fi, open(dst, "w") as fo:
    for line in fi:
        row = json.loads(line)
        if row["id"].startswith("mbppt-") or row["id"].startswith("dom-mbpp"):
            dropped += 1
            continue
        fo.write(json.dumps(row, ensure_ascii=False) + "\n")
        kept += 1
print(f"kept {kept}  dropped(eval-dataset overlap) {dropped}")
PYEOF

PYTHONPATH=. python3 -u scripts/regenerate_train_data.py \
  --model /kl_infra_infer_intern/wangzhuoyu/TAPS-SP/models/Qwen3-4B \
  --server-address 127.0.0.1:30000 127.0.0.1:30010 \
                   127.0.0.1:30020 127.0.0.1:30030 \
                   127.0.0.1:30040 127.0.0.1:30050 \
                   127.0.0.1:30060 127.0.0.1:30070 \
  --concurrency 64 \
  --max-tokens 2048 \
  --temperature 0 \
  --reasoning disable \
  --input-file-path "$IN" \
  --output-file-path "$OUT" \
  --resume
