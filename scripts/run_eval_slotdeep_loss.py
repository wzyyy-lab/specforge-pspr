"""Export and evaluate the predeclared step1000 SlotDeep loss pilots.

Uses the unchanged decode_lattice.py and existing fixed HF teacher-prefix
diagnostic. It does not change gates, choose checkpoints, or train a model.
Launch only AFTER all three training arms have exited and GPUs are free.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[1]
TAPS = ROOT.parent / "TAPS-SP"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def logged(command, cwd, env, log):
    print("EVAL_COMMAND", shlex.join(command), flush=True)
    with log.open("x") as sink:
        result = subprocess.run(command, cwd=cwd, env=env, stdout=sink,
                                stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        raise RuntimeError(f"command exit={result.returncode}; see {log}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", default="0,1,2")
    parser.add_argument("--arms", default="t0,t1,t2", help="explicit fixed-step arms; default original three-arm suite")
    parser.add_argument("--plan", action="store_true")
    args = parser.parse_args()
    gpus = args.gpus.split(",")
    arms = args.arms.split(",")
    if len(set(arms)) != len(arms) or not set(arms) <= {"t0", "t1", "t2", "t3"}:
        parser.error("need distinct supported arms: t0,t1,t2,t3")
    if len(gpus) != len(arms) or len(set(gpus)) != len(gpus) or not all(g.isdigit() for g in gpus):
        parser.error("need one distinct numeric GPU ID per arm")
    jobs = []
    for arm, gpu in zip(arms, gpus):
        run = f"qwen3-4b-pspr-slotdeep-loss-{arm}-20260905"
        ckpt = ROOT / f"outputs/{run}/{run}-step1000/training_state.pt"
        export = ROOT / f"outputs/{run}_step1000_decode"
        tag = f"SLOTDEEP_LOSS_{arm.upper()}_20260905"
        decoder = ["python", "-u", "scripts/decode_lattice.py",
            "--target-model", "models/Qwen3-4B", "--draft-model", str(export / "backbone"),
            "--lattice-head", str(export / "selector.pt"), "--modes", "latgate",
            "--datasets", "gsm8k,math500,humaneval,mbpp,alpaca,mt-bench",
            "--gate-tau", "0", "--gate-rho", "3", "--gate-theta", "0", "--gate-stats",
            "--eval-reserved", "--max-samples", "20", "--max-new-tokens", "256",
            "--shuffle-seed", "2026", "--json-output", str(ROOT / f"outputs/EVAL_{tag}.json")]
        exporter = ["python", "scripts/export_pspr_cloze_for_decode.py", "--checkpoint", str(ckpt),
            "--draft-config", "configs/qwen3-4b-pspr-slotdeep.json", "--backbone-template",
            str(TAPS / "models/Qwen3-4B-DFlash-b16"), "--verify-frozen",
            str(TAPS / "models/Qwen3-4B-DFlash-b16"), "--verify-roundtrip", "--out", str(export)]
        diagnostic = ["python", "-u", "scripts/measure_ranker_teacher_forced.py",
            "--features", "cache/hidden_states/slotdeep_eval_20260905",
            "--draft-config", "configs/qwen3-4b-pspr-slotdeep.json", "--checkpoint", str(ckpt),
            "--target-model", str(TAPS / "models/Qwen3-4B"), "--num-samples", "120",
            "--num-anchors", "512", "--chunk-blocks", "64", "--max-length", "3072",
            "--seed", "2026", "--gate-rho", "3", "--gate-tau", "0", "--gate-theta", "0",
            "--preserve-selector-fp32", "--per-sample-diagnostics",
            "--dump-decisions", f"outputs/RISK_{tag}.npz", "--include-base-decisions",
            "--json-output", f"outputs/RISK_{tag}.json"]
        jobs.append(dict(arm=arm, gpu=gpu, checkpoint=str(ckpt), export=str(export),
                         tag=tag, commands=[exporter, decoder, diagnostic]))
    if args.plan:
        print(json.dumps(jobs, indent=2))
        return
    usage = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used",
                                    "--format=csv,noheader,nounits"], text=True)
    memory = {k.strip(): int(v.strip()) for k, v in (line.split(",") for line in usage.splitlines())}
    if any(g not in memory or memory[g] >= 500 for g in gpus):
        raise RuntimeError(f"evaluation GPU(s) not free: {memory}")
    for job in jobs:
        if not Path(job["checkpoint"]).is_file():
            raise FileNotFoundError(job["checkpoint"])
        outputs = [Path(job["export"])]
        outputs += [ROOT / f"outputs/{p}_{job['tag']}{suffix}"
                    for p, suffix in (("EXPORT", ".log"), ("EVAL", ".log"), ("EVAL", ".json"),
                                      ("TF", ".log"), ("RISK", ".json"), ("RISK", ".npz"),
                                      ("EVAL", "_PROVENANCE.json"))]
        if any(p.exists() for p in outputs):
            raise FileExistsError(f"refusing to overwrite evaluation outputs: {outputs}")
    common = os.environ.copy()
    common.update(PYTHONPATH=".", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4",
                  HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", no_proxy="localhost,127.0.0.1",
                  NO_PROXY="localhost,127.0.0.1")
    source_paths = [TAPS / "scripts/decode_lattice.py", TAPS / "model/utils.py",
                    ROOT / "specforge/modeling/draft/pspr_slotdeep.py",
                    ROOT / "specforge/modeling/draft/pspr_cloze.py",
                    ROOT / "scripts/measure_ranker_teacher_forced.py",
                    ROOT / "scripts/export_pspr_cloze_for_decode.py", Path(__file__).resolve()]
    source_paths += sorted((TAPS / "models/Qwen3-4B").glob("*.safetensors"))
    source_paths += [TAPS / "models/Qwen3-4B-DFlash-b16/model.safetensors"]
    hashes = {str(p): sha(p) for p in source_paths}
    def run(job):
        env = {**common, "CUDA_VISIBLE_DEVICES": job["gpu"]}
        logged(job["commands"][0], ROOT, env, ROOT / f"outputs/EXPORT_{job['tag']}.log")
        manifest = dict(created_utc=datetime.now(timezone.utc).isoformat(), **job,
            source_and_model_sha256=hashes, checkpoint_sha256=sha(job["checkpoint"]),
            selector_sha256=sha(Path(job["export"]) / "selector.pt"),
            scope="Fixed six-domain development benchmark with known MATH500 index7 overlap; separate fixed HF teacher-prefix proxy, not exact decode replay; no speed or task-accuracy claim")
        with (ROOT / f"outputs/EVAL_{job['tag']}_PROVENANCE.json").open("x") as sink:
            json.dump(manifest, sink, indent=2)
        logged(job["commands"][1], TAPS, env, ROOT / f"outputs/EVAL_{job['tag']}.log")
        logged(job["commands"][2], ROOT, env, ROOT / f"outputs/TF_{job['tag']}.log")
        print("LOSS_ARM_EVAL_DONE", job["arm"], flush=True)
    with ThreadPoolExecutor(max_workers=len(jobs)) as executor:
        list(executor.map(run, jobs))


if __name__ == "__main__":
    main()
