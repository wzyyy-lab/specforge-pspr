"""Fresh random, matched-H15 evaluation; shard manifests, no model/code edits."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
from queue import Empty, Queue
import subprocess
import sys
from threading import Event

from run_stage1_domino_h15_eval50 import (DATASETS, DOMINO, ROOT, S2, SF, TAPS,
                                        TARGET, digest, save_exclusive, timestamp)

WORK = SF / "outputs/STAGE2_DOMINO_H15_EVAL200_20260907"
MANIFEST = WORK / "prompts.json"
PRIOR = SF / "outputs/STAGE1_STAGE2_DOMINO_H15_EVAL50_20260907"
SEED = 20260907
REQUESTED = 200
EXPECTED = dict(gsm8k=200, math500=200, humaneval=164, mbpp=200, alpaca=200, **{"mt-bench": 80})
sys.path.insert(0, str(TAPS))


def env_for(gpu):
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(gpu), PYTHONPATH=".", OMP_NUM_THREADS="4",
        MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", TOKENIZERS_PARALLELISM="false",
        HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1")
    return env


def gpu_memory():
    rows = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used",
        "--format=csv,noheader,nounits"], text=True).strip().splitlines()
    return {int(row.split(",")[0]): int(row.split(",")[1]) for row in rows}


def prepare():
    from model import load_and_process_dataset
    from scripts.eval_protocol import read_manifest
    WORK.mkdir(parents=True, exist_ok=True)
    (WORK / "manifests").mkdir(exist_ok=True)
    if MANIFEST.exists():
        raise FileExistsError("Prepared manifest exists; do not redraw or overwrite it")
    previous = json.loads((SF / "outputs/STAGE2_EVAL50_REGEN8B_20260906/prompts50.json").read_text())
    old_hashes = {r["prompt_sha256"] for r in previous["prompts"]}
    selected, info, shards = [], {}, []
    for ds in DATASETS:
        data = load_and_process_dataset(ds)
        fingerprint = getattr(data, "_fingerprint", None)
        unique_population = len({r["turns"][0] for r in data})
        shuffled = data.add_column("evaluation_source_index", list(range(len(data)))).shuffle(seed=SEED)
        seen, chosen = set(), []
        for row in shuffled:
            content = row["turns"][0]
            key = hashlib.sha256(content.encode()).hexdigest()
            if key in seen:
                continue
            seen.add(key)
            chosen.append(dict(dataset=ds, prompt_index=len(chosen), content=content,
                prompt_sha256=key, source_row_index=int(row["evaluation_source_index"]),
                metadata={k: row[k] for k in ("category", "subject", "level", "task_id", "prompt_id", "unique_id") if k in row}))
            if len(chosen) == REQUESTED:
                break
        assert len(chosen) == EXPECTED[ds] == min(REQUESTED, unique_population)
        info[ds] = dict(population_rows=len(data), unique_first_prompts=unique_population,
            requested=REQUESTED, evaluated=len(chosen), fingerprint=fingerprint,
            old50_overlap=sum(r["prompt_sha256"] in old_hashes for r in chosen),
            sampling="first unique prompts after datasets.shuffle(seed); census when population<200")
        for shard_id, start in enumerate(range(0, len(chosen), 50)):
            part = [dict(r, prompt_index=i, canonical_prompt_index=r["prompt_index"])
                    for i, r in enumerate(chosen[start:start + 50])]
            path = WORK / "manifests" / f"{ds}_{shard_id}.json"
            save_exclusive(path, dict(version=1, shuffle_seed=SEED, max_samples=len(part),
                heldout_claim=False, prompts=part))
            read_manifest(path, [ds], len(part), SEED)
            shards.append(dict(dataset=ds, shard=shard_id, count=len(part), path=str(path)))
        selected.extend(chosen)
        print("SAMPLE", ds, json.dumps(info[ds]), flush=True)
    save_exclusive(MANIFEST, dict(version=2, shuffle_seed=SEED, requested_per_domain=REQUESTED,
        first_turn_only=True, heldout_claim=False,
        limitation="Previously used development benchmarks; random sample is not a decontamination or new-test guarantee. No duplicated padding to200.",
        datasets=info, shards=shards, prompts=selected))


def check_manifest():
    from scripts.eval_protocol import read_manifest
    manifest = json.loads(MANIFEST.read_text())
    assert manifest["shuffle_seed"] == SEED
    expected = {(r["dataset"], r["prompt_index"]): r for r in manifest["prompts"]}
    assert len(expected) == sum(EXPECTED.values())
    visited = set()
    for shard in manifest["shards"]:
        rows = read_manifest(shard["path"], [shard["dataset"]], shard["count"], SEED)[shard["dataset"]]
        for r in rows:
            key = (r["dataset"], r["canonical_prompt_index"])
            assert key not in visited and expected[key]["prompt_sha256"] == r["prompt_sha256"]
            visited.add(key)
    assert visited == set(expected)
    return manifest


def command(method, shard, output, max_new=256):
    common = ["--target-model", str(TARGET), "--datasets", shard["dataset"],
        "--max-samples", str(shard["count"]), "--max-new-tokens", str(max_new),
        "--shuffle-seed", str(SEED), "--prompt-manifest", str(shard["path"]),
        "--draft-attn", "sdpa", "--json-output", str(output)]
    if method == "stage2":
        return ["python", "-u", "scripts/decode_lattice.py", *common,
            "--draft-model", str(S2 / "backbone"), "--lattice-head", str(S2 / "selector.pt"),
            "--modes", "oneshot,latgate", "--gate-rho", "3", "--gate-tau", "0", "--gate-theta", "0",
            "--gate-stats", "--stop-policy", "official"]
    return ["python", "-u", "scripts/bench_official_domino.py", *common,
        "--domino-draft", str(DOMINO), "--max-proposals", "15"]


def collect_hashes():
    paths = [Path(__file__), Path(__file__).with_name("run_stage1_domino_h15_eval50.py"), MANIFEST,
        TAPS / "scripts/decode_lattice.py", TAPS / "scripts/bench_official_domino.py",
        TAPS / "scripts/domino_proposal_cap.py", TAPS / "scripts/eval_protocol.py",
        TAPS / "model/utils.py", TAPS / "model/dflash.py",
        SF / "specforge/modeling/draft/pspr_slotdeep.py", S2 / "selector.pt"]
    paths += list((WORK / "manifests").glob("*.json"))
    for directory in (TARGET, DOMINO, S2 / "backbone"):
        for suffix in ("*.safetensors", "*.json", "*.py", "merges.txt"):
            paths += list(directory.glob(suffix))
    hashes = {str(path): digest(path) for path in sorted(set(paths))}
    prior_hashes = json.loads((PRIOR / "launch_provenance.json").read_text())["hashes"]
    for path in hashes.keys() & prior_hashes.keys():
        assert hashes[path] == prior_hashes[path], ("Prior evaluated input changed", path)
    return hashes


def preflight():
    manifest = check_manifest()
    hashes = collect_hashes()
    record = dict(timestamp=timestamp(), status="PASS_CPU_ONLY", hashes=hashes,
        counts={ds: manifest["datasets"][ds]["evaluated"] for ds in DATASETS},
        gpu_memory=gpu_memory(), total_prompts=len(manifest["prompts"]), seed=SEED)
    save_exclusive(WORK / "preflight.json", record)
    print("EVAL200_PREFLIGHT_PASS_CPU_ONLY", record["counts"], flush=True)


def run_job(method, shard, gpu, directory, max_new=256):
    name = f"{method}_{shard['dataset']}_{shard['shard']}"
    output = directory / f"{name}.json"
    log = directory / f"{name}.log"
    for path in (output, log, directory / f"{name}.launch.json", directory / f"{name}.exit.json"):
        if path.exists():
            raise FileExistsError(path)
    argv = command(method, shard, output, max_new)
    save_exclusive(directory / f"{name}.launch.json", dict(timestamp=timestamp(),
        method=method, shard=shard, gpu=gpu, cwd=str(TAPS), argv=argv))
    print("START", name, "GPU", gpu, "N", shard["count"], flush=True)
    with log.open("x") as stream:
        code = subprocess.call(argv, cwd=TAPS, env=env_for(gpu), stdout=stream, stderr=subprocess.STDOUT)
    record = dict(name=name, method=method, shard=shard, gpu=gpu, exit_code=code,
        timestamp=timestamp(), result=str(output), result_sha256=digest(output) if output.exists() else None)
    save_exclusive(directory / f"{name}.exit.json", record)
    print("EXIT", name, code, flush=True)
    return record


def witness():
    assert json.loads((WORK / "preflight.json").read_text())["status"] == "PASS_CPU_ONLY"
    assert gpu_memory()[7] < 500
    directory = WORK / "witness"
    directory.mkdir(exist_ok=True)
    row = next(r for r in check_manifest()["prompts"] if r["dataset"] == "gsm8k")
    path = directory / "prompt.json"
    save_exclusive(path, dict(version=1, shuffle_seed=SEED, max_samples=1, prompts=[dict(row, prompt_index=0)]))
    shard = dict(dataset="gsm8k", shard=0, count=1, path=str(path))
    records = [run_job(method, shard, 7, directory, 64) for method in ("stage2", "domino")]
    assert all(r["exit_code"] == 0 for r in records), records
    for record in records:
        data = json.loads(Path(record["result"]).read_text())
        assert data["proposal_slots"] == 15
        assert len(data["results"]) == (2 if record["method"] == "stage2" else 1)
        assert all(r["prompt_sha256"] == row["prompt_sha256"] and r["num_blocks"] > 0 and
                   all(1 <= a <= 16 for a in r["acceptance_lengths"]) for r in data["results"])
    save_exclusive(WORK / "witness.json", dict(status="PASS", records=records, timestamp=timestamp()))
    print("EVAL200_GPU_WITNESS_PASS", flush=True)


def run():
    assert json.loads((WORK / "witness.json").read_text())["status"] == "PASS"
    memory = gpu_memory()
    assert len(memory) == 8 and all(v < 500 for v in memory.values()), memory
    manifest = check_manifest()
    hashes = collect_hashes()
    assert hashes == json.loads((WORK / "preflight.json").read_text())["hashes"]
    save_exclusive(WORK / "launch_provenance.json", dict(timestamp=timestamp(), hashes=hashes,
        gpu_memory=memory, proposal_slots=15, max_new_tokens=256, seed=SEED,
        stage2=str(S2), domino=str(DOMINO), modes=["stage2_oneshot", "stage2_latgate", "domino_h15"],
        counts=EXPECTED, training_changed=False))
    tasks = [(method, shard) for shard in manifest["shards"] for method in ("stage2", "domino")]
    tasks.sort(key=lambda x: x[1]["count"] * (2.5 if x[0] == "stage2" else 1) *
        (2 if x[1]["dataset"] in ("alpaca", "mt-bench") else 1), reverse=True)
    queue, failed = Queue(), Event()
    for item in tasks:
        queue.put(item)
    def worker(gpu):
        results = []
        while not failed.is_set():
            try:
                method, shard = queue.get_nowait()
            except Empty:
                break
            record = run_job(method, shard, gpu, WORK)
            results.append(record)
            if record["exit_code"]:
                failed.set()
        return results
    with ThreadPoolExecutor(max_workers=8) as pool:
        groups = list(pool.map(worker, range(8)))
    results = sum(groups, [])
    unchanged = collect_hashes() == hashes
    status = "DONE" if len(results) == len(tasks) and not failed.is_set() and unchanged else "FAILED"
    save_exclusive(WORK / "completion.json", dict(status=status, results=results,
        tasks_expected=len(tasks), tasks_completed=len(results), input_hashes_unchanged=unchanged,
        timestamp=timestamp()))
    print("EVAL200_" + status, flush=True)
    if status != "DONE":
        raise RuntimeError("Evaluation incomplete; inspect preserved artifacts, no silent retry")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "preflight", "witness", "run"))
    args = parser.parse_args()
    {"prepare": prepare, "preflight": preflight, "witness": witness, "run": run}[args.action]()
