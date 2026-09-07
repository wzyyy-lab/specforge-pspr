"""Inference-only matched-H15 rerun; fresh exclusive artifacts, no training."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
SF = ROOT / "SpecForge"
TAPS = ROOT / "TAPS-SP"
OLD = SF / "outputs/STAGE2_EVAL50_REGEN8B_20260906"
WORK = SF / "outputs/STAGE1_STAGE2_DOMINO_H15_EVAL50_20260907"
S1 = SF / "outputs/qwen3-4b-pspr-slotdeep-loss-t3-20260905_step1000_decode"
S2 = SF / "outputs/qwen3-4b-pspr-slotdeep-stage2-20260906_step9052_decode"
DOMINO = TAPS / "models/Qwen3-4B-Domino-b16"
TARGET = TAPS / "models/Qwen3-4B"
PROMPTS = OLD / "prompts50.json"
DATASETS = ["gsm8k", "math500", "humaneval", "mbpp", "alpaca", "mt-bench"]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def save_exclusive(path, payload):
    with path.open("x") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")


def make_jobs():
    jobs = []

    def common(domains):
        return ["--target-model", str(TARGET), "--datasets", ",".join(domains),
            "--max-samples", "50", "--max-new-tokens", "256", "--shuffle-seed", "2026",
            "--prompt-manifest", str(PROMPTS), "--draft-attn", "sdpa"]

    for gpu, ds in enumerate(DATASETS):
        name = f"stage1_t3_{ds}"
        command = ["python", "-u", "scripts/decode_lattice.py", *common([ds]),
            "--draft-model", str(S1 / "backbone"), "--lattice-head", str(S1 / "selector.pt"),
            "--modes", "oneshot,latgate", "--gate-tau", "0", "--gate-rho", "3",
            "--gate-theta", "0", "--gate-stats", "--stop-policy", "official",
            "--json-output", str(WORK / f"{name}.json")]
        jobs.append(dict(name=name, gpu=gpu, argv=command))
    for gpu, domains in ((6, DATASETS[:4]), (7, DATASETS[4:])):
        name = f"domino_h15_gpu{gpu}"
        command = ["python", "-u", "scripts/bench_official_domino.py", *common(domains),
            "--domino-draft", str(DOMINO), "--max-proposals", "15",
            "--json-output", str(WORK / f"{name}.json")]
        jobs.append(dict(name=name, gpu=gpu, argv=command))
    return jobs


def main():
    witness = json.loads((WORK / "domino_h15_witness.json").read_text())
    assert witness["status"] == "PASS"
    assert witness["cap15_verifier_input_width"] == 16
    assert digest(PROMPTS) == "c2b11f8c25a2e65371b5e4873b461636825c0c06f79db74fefa9120a8ecec6c0"
    gpu_rows = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used",
        "--format=csv,noheader,nounits"], text=True).strip().splitlines()
    assert len(gpu_rows) == 8 and all(int(row.split(",")[1]) < 500 for row in gpu_rows), gpu_rows
    jobs = make_jobs()
    for job in jobs:
        for suffix in (".json", ".log", ".launch.json", ".exit.json"):
            if (WORK / (job["name"] + suffix)).exists():
                raise FileExistsError(WORK / (job["name"] + suffix))
    paths = [Path(__file__), PROMPTS, OLD / "slotdeep_stage2_step9052.json",
        TAPS / "scripts/decode_lattice.py", TAPS / "scripts/bench_official_domino.py",
        TAPS / "scripts/domino_proposal_cap.py", TAPS / "scripts/eval_protocol.py",
        TAPS / "model/dflash.py", SF / "specforge/modeling/draft/pspr_slotdeep.py",
        S1 / "selector.pt", S2 / "selector.pt"]
    for directory in (TARGET, DOMINO, S1 / "backbone", S2 / "backbone"):
        paths += list(directory.glob("*.safetensors"))
        paths += list(directory.glob("*.json"))
        paths += list(directory.glob("*.py"))
    paths = sorted(set(paths))
    hashes = {str(path): digest(path) for path in paths}
    old_preflight = json.loads((OLD / "preflight.json").read_text())
    for path in (TAPS / "scripts/decode_lattice.py", TAPS / "scripts/eval_protocol.py",
                 TAPS / "model/dflash.py", SF / "specforge/modeling/draft/pspr_slotdeep.py"):
        assert hashes[str(path)] == old_preflight["source_code_sha256"][str(path)]
    t3_provenance = json.loads((SF / "outputs/EVAL_SLOTDEEP_LOSS_T3_20260905_PROVENANCE.json").read_text())
    assert hashes[str(S1 / "selector.pt")] == t3_provenance["selector_sha256"]
    for path in TARGET.glob("*.safetensors"):
        assert hashes[str(path)] == t3_provenance["source_and_model_sha256"][str(path)]
    official_identity = json.loads((OLD / "official_domino_identity.json").read_text())
    assert hashes[str(DOMINO / "model.safetensors")] == official_identity["model.safetensors"]["cached_official_lfs_sha256"]
    assert hashes[str(S2 / "selector.pt")] == "9d8ff476e5e5ab0cc36942490cc8abd2c8e711626b280c04a39137a20d3d38b7"
    assert hashes[str(S2 / "backbone/model.safetensors")] == "41e770ccfb5f0c31616d535141622bac2a8ef4ce9da3e3c93729802296f13610"
    save_exclusive(WORK / "launch_provenance.json", dict(timestamp=timestamp(),
        hashes=hashes, gpu_memory_before=gpu_rows, jobs=jobs,
        reuse_stage2=str(OLD / "slotdeep_stage2_step9052.json"),
        stage1_definition="best T3: frozen official DFlash, S1@7115 + selector continuation1000",
        same_h=15, acceptance_includes_bonus=True, samples_per_domain=50,
        policy="greedy/nonthinking/BF16/SDPA/256 new tokens/official committed EOS"))

    def run(job):
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=str(job["gpu"]), PYTHONPATH=".",
            OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4",
            TOKENIZERS_PARALLELISM="false", HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1")
        save_exclusive(WORK / (job["name"] + ".launch.json"),
            dict(**job, cwd=str(TAPS), timestamp=timestamp()))
        print("START", job["name"], "GPU", job["gpu"], flush=True)
        with (WORK / (job["name"] + ".log")).open("x") as log:
            process = subprocess.Popen(job["argv"], cwd=TAPS, env=env,
                stdout=log, stderr=subprocess.STDOUT)
            code = process.wait()
        result_path = WORK / (job["name"] + ".json")
        record = dict(name=job["name"], exit_code=code, timestamp=timestamp(),
            result_sha256=digest(result_path) if result_path.exists() else None)
        save_exclusive(WORK / (job["name"] + ".exit.json"), record)
        print("EXIT", record, flush=True)
        return record

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(run, job) for job in jobs]
        results = [future.result() for future in as_completed(futures)]
    unchanged = all(digest(path) == expected for path, expected in hashes.items())
    status = "DONE" if unchanged and all(r["exit_code"] == 0 for r in results) else "FAILED"
    save_exclusive(WORK / "completion.json", dict(status=status, results=results,
        input_and_code_hashes_unchanged=unchanged, timestamp=timestamp()))
    print("MATCHED_H15_" + status, flush=True)
    if status != "DONE":
        raise RuntimeError("Evaluation failed or inputs changed; inspect preserved logs")


if __name__ == "__main__":
    main()
