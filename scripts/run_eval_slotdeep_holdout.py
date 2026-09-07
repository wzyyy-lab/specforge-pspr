"""Run five fixed native policies on the predeclared new PB validation prompts.

No gate or checkpoint selection is performed. Never uses calibration labels.
Run only when the requested GPUs are free and all exports are complete.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess

from scripts.run_eval_slotdeep_loss import ROOT, logged, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", default="3,4,5,6,7")
    parser.add_argument("--plan", action="store_true")
    args = parser.parse_args()
    gpus = args.gpus.split(",")
    if len(gpus) != 5 or len(set(gpus)) != 5 or not all(g.isdigit() for g in gpus):
        parser.error("need five distinct numeric GPU IDs")
    exports = [("CLOZE7115", "qwen3-4b-pspr-cloze_step7115_decode"),
               ("SLOTDEEP7115", "qwen3-4b-pspr-slotdeep-s1-20260905_step7115_decode")]
    exports += [(f"LOSS_T{i}", f"qwen3-4b-pspr-slotdeep-loss-t{i}-20260905_step1000_decode")
                for i in range(3)]
    prompts = ROOT / "cache/dataset/pspr_holdout_20260905/validation.jsonl"
    jobs = []
    for (label, folder), gpu in zip(exports, gpus):
        export = ROOT / "outputs" / folder
        tag = f"PB_VALIDATION_{label}_20260905"
        result = ROOT / f"outputs/{tag}.json"
        command = ["python", "-u", "scripts/evaluate_pspr_holdout.py", "--prompts", str(prompts),
                   "--export", str(export), "--output", str(result),
                   "--max-new-tokens", "256", "--max-prompt-tokens", "2816"]
        jobs.append(dict(label=label, gpu=gpu, export=str(export), command=command,
                         result=str(result), log=str(ROOT / f"outputs/{tag}.log"),
                         provenance=str(ROOT / f"outputs/{tag}_PROVENANCE.json")))
    if args.plan:
        print(json.dumps(jobs, indent=2))
        return
    usage = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used",
                                    "--format=csv,noheader,nounits"], text=True)
    memory = {k.strip(): int(v.strip()) for k, v in (line.split(",") for line in usage.splitlines())}
    if any(g not in memory or memory[g] >= 500 for g in gpus):
        raise RuntimeError(f"holdout GPU(s) not free: {memory}")
    assert prompts.is_file()
    for job in jobs:
        for name in ("selector.pt", "backbone/model.safetensors"):
            if not (Path(job["export"]) / name).is_file():
                raise FileNotFoundError(Path(job["export"]) / name)
        for field in ("result", "log", "provenance"):
            if Path(job[field]).exists():
                raise FileExistsError(job[field])
    common = os.environ.copy()
    common.update(PYTHONPATH=".", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4",
                  HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", no_proxy="localhost,127.0.0.1",
                  NO_PROXY="localhost,127.0.0.1")
    sources = [Path(__file__).resolve(), ROOT / "scripts/evaluate_pspr_holdout.py",
               ROOT / "scripts/dump_eval_traces.py", ROOT.parent / "TAPS-SP/scripts/decode_lattice.py",
               prompts, prompts.parent / "manifest.json"]
    sources += sorted((ROOT.parent / "TAPS-SP/models/Qwen3-4B").glob("*.safetensors"))
    hashes = {str(p): sha(p) for p in sources}
    def run(job):
        export = Path(job["export"])
        manifest = dict(created_utc=datetime.now(timezone.utc).isoformat(), **job,
            source_and_input_sha256=hashes, selector_sha256=sha(export / "selector.pt"),
            backbone_sha256=sha(export / "backbone/model.safetensors"),
            scope="Single training seed; fixed native policies on predeclared new PB validation split. Exact normalized-prompt exclusion from declared head-training files, not semantic or pretraining decontamination. Target agreement, not task accuracy or speed.")
        with Path(job["provenance"]).open("x") as sink:
            json.dump(manifest, sink, indent=2)
        logged(job["command"], ROOT, {**common, "CUDA_VISIBLE_DEVICES": job["gpu"]}, Path(job["log"]))
        print("PB_VALIDATION_DONE", job["label"], flush=True)
    with ThreadPoolExecutor(max_workers=5) as executor:
        list(executor.map(run, jobs))


if __name__ == "__main__":
    main()
