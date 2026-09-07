"""Fixed, bounded calibration experiment jobs; never train or alter model weights.

Use --plan to inspect commands. Every phase is explicit; no outcome-dependent
threshold sweep or silent resume. Only four GPU workers can run in this runner.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time

from scripts.run_eval_slotdeep_loss import ROOT, TAPS, logged, sha

OUT = ROOT / "outputs/slotdeep_calibration_20260906"
OLD = ROOT / "cache/dataset/pspr_holdout_20260905"
NEW = ROOT / "cache/dataset/pspr_confirmation_20260906"
ARTIFACT = OUT / "calibration.json"
RUNS = dict(CE="qwen3-4b-pspr-slotdeep-turn-s1-20260905",
            KD="qwen3-4b-pspr-slotdeep-distill-s1-20260906")
OLD_RESULTS = dict(CE=ROOT / "outputs/PB_VALIDATION_TURN_S1_20260905.json",
                   KD=ROOT / "outputs/PB_VALIDATION_DISTILL_S1_20260906.json")
DOMAINS = "gsm8k,math500,humaneval,mbpp,alpaca,mt-bench"


def export_path(arm):
    return ROOT / "outputs" / (RUNS[arm] + "_step7115_decode")


def jobs_for(phase):
    jobs = []
    arms = [(a, c) for a in RUNS for c in (False, True)] if phase == "confirmation" else [(a, phase in {"six", "gate-smoke"}) for a in RUNS]
    for arm, calibrated in arms:
        tag = f"{phase}_{arm}" + ("_calibrated" if calibrated else "_native")
        result = OUT / (tag + ".json")
        export = export_path(arm)
        base = ["python", "-u", "scripts/evaluate_pspr_holdout.py", "--export", str(export),
                "--prompts", str((NEW if phase == "confirmation" else OLD) / "validation.jsonl"),
                "--max-new-tokens", "256", "--max-prompt-tokens", "2816"]
        if phase == "smoke":
            command = base + ["--max-samples", "2", "--output", str(result)]
        elif phase in {"diagnostics", "gate-smoke"}:
            run = RUNS[arm]
            command = ["python", "-u", "scripts/measure_ranker_teacher_forced.py",
                       "--features", str(ROOT / "cache/hidden_states/pspr_holdout_20260905/calibration"),
                       "--draft-config", "configs/qwen3-4b-pspr-slotdeep.json",
                       "--checkpoint", str(ROOT / f"outputs/{run}/{run}-step7115/training_state.pt"),
                       "--target-model", str(TAPS / "models/Qwen3-4B"),
                       "--num-samples", "512", "--num-anchors", "512", "--chunk-blocks", "64",
                       "--max-length", "3072", "--seed", "2026", "--gate-rho", "3",
                       "--gate-tau", "0", "--gate-theta", "0", "--preserve-selector-fp32",
                       "--per-sample-diagnostics", "--include-base-decisions",
                       "--dump-decisions", str(result.with_suffix(".npz")), "--json-output", str(result)]
            if phase == "gate-smoke":
                command[command.index('--num-samples')+1] = '2'
        elif phase == "confirmation":
            command = base + ["--gate-stats", "--output", str(result)]
            if calibrated:
                command += ["--calibration", str(ARTIFACT), "--calibration-arm", arm]
        elif phase == "six":
            command = ["python", "-u", "scripts/decode_lattice.py",
                       "--target-model", str(TAPS / "models/Qwen3-4B"),
                       "--draft-model", str(export / "backbone"), "--lattice-head", str(export / "selector.pt"),
                       "--modes", "latgate", "--datasets", DOMAINS, "--gate-tau", "0",
                       "--gate-theta", "0", "--gate-stats", "--allow-policy-mismatch",
                       "--eval-reserved", "--max-samples", "20", "--max-new-tokens", "256",
                       "--shuffle-seed", "2026", "--json-output", str(result)]
        else:
            raise ValueError(phase)
        jobs.append(dict(arm=arm, tag=tag, result=str(result), export=str(export), command=command,
                         cwd=str(TAPS if phase == "six" else ROOT), calibrated=calibrated))
    return jobs


def write_json(path, value):
    with Path(path).open("x") as sink:
        json.dump(value, sink, indent=2)
        sink.write("\n")


def require_success(tag):
    result = json.loads((OUT / f'{tag}_EXIT.json').read_text())
    assert result['success'] is True, f'prerequisite did not succeed: {tag}'
    provenance = json.loads((OUT / f'{tag}_PROVENANCE.json').read_text())
    assert all(sha(p) == h for p, h in provenance['source_sha256'].items()), f'stale prerequisite inputs: {tag}'
    assert sha(Path(provenance['export']) / 'selector.pt') == provenance['selector_sha256']


def prepare_six_prompts():
    # Same loader, reserved count, shuffle and slice as the unchanged CLI.
    from scripts.dump_eval_traces import capture as _capture
    import scripts.decode_lattice as reference
    records = []
    for domain in DOMAINS.split(","):
        dataset = reference.load_and_process_dataset(domain)
        dataset = reference.select_dataset_samples(dataset,
            max_samples=20 if domain in {"humaneval", "mt-bench"} else 40,
            sample_offset=0, shuffle_seed=2026)
        for index in range(min(len(dataset), 20)):
            records.append(dict(id=f"{domain}:{index}", source=domain, turns=[dataset[index]["turns"][0]]))
    assert len(records) == 120
    path = OUT / "six_prompt_identity.jsonl"
    with path.open("x") as sink:
        for row in records:
            sink.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("smoke", "diagnostics", "gate-smoke", "confirmation", "six"), required=True)
    parser.add_argument("--gpus", required=True)
    parser.add_argument("--plan", action="store_true")
    args = parser.parse_args()
    jobs = jobs_for(args.phase)
    gpus = args.gpus.split(",")
    if len(gpus) != len(jobs) or len(set(gpus)) != len(gpus) or not all(g.isdigit() for g in gpus):
        parser.error("one distinct numeric GPU ID per job, at most four")
    for job, gpu in zip(jobs, gpus):
        job["gpu"] = gpu
    if args.plan:
        print(json.dumps(jobs, indent=2))
        return
    if args.phase != 'smoke':
        for arm in RUNS:
            require_success(f'smoke_{arm}_native')
    if args.phase in {'confirmation', 'six'}:
        for arm in RUNS:
            require_success(f'gate-smoke_{arm}_calibrated')
    usage = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used",
                                    "--format=csv,noheader,nounits"], text=True)
    memory = {k.strip(): int(v.strip()) for k, v in (line.split(",") for line in usage.splitlines())}
    assert all(g in memory and memory[g] < 500 for g in gpus), f"GPUs not free: {memory}"
    OUT.mkdir(parents=True, exist_ok=True)
    for job in jobs:
        for suffix in (".json", ".npz", ".log", "_PROVENANCE.json", "_EXIT.json", "_stats.json", "_stats.log"):
            if (OUT / (job["tag"] + suffix)).exists():
                raise FileExistsError(OUT / (job["tag"] + suffix))
        assert (Path(job["export"]) / "selector.pt").is_file()
    env = {**os.environ, "PYTHONPATH": ".", "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4",
           "OPENBLAS_NUM_THREADS": "4", "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
           "no_proxy": "localhost,127.0.0.1", "NO_PROXY": "localhost,127.0.0.1"}
    os.environ.update({k: env[k] for k in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE")})
    if args.phase in {"six", "confirmation", "gate-smoke"}:
        from scripts.calibrate_pspr_gate import validate_for_evaluation
        prompts = (prepare_six_prompts() if args.phase == 'six' else
                   (NEW if args.phase == 'confirmation' else OLD) / 'validation.jsonl')
        validated = {}
        # Seal BOTH rules before ANY confirmation worker can read outcomes.
        # gate-smoke subsequently uses calibration features, never confirmation.
        import torch
        from types import SimpleNamespace
        from scripts.dump_eval_traces import capture as _capture
        import scripts.decode_lattice as reference
        for arm in RUNS:
            export = export_path(arm)
            policy, provenance = validate_for_evaluation(ARTIFACT, arm, export,
                TAPS / 'models/Qwen3-4B', prompts, TAPS / 'scripts/decode_lattice.py')
            payload = torch.load(export / 'selector.pt', map_location='cpu', weights_only=False, mmap=True)
            p = payload['decode_policy']
            assert p == provenance['original_policy']
            reference.validate_selector_policy(payload, SimpleNamespace(
                allow_policy_mismatch=False, beam=1, path_branch=1, repair_margin=0.,
                gate_rho=p['selector_gate_rho'], gate_tau=p['selector_gate_tau'],
                gate_theta=p['selector_gate_theta'], gate_skip0=p['selector_gate_skip_first']), {'latgate'})
            validated[arm] = (policy, provenance)
        for job in jobs:
            policy, provenance = validated[job['arm']]
            job['sealed_calibration_comparison'] = {a: v[1] for a, v in validated.items()}
            if job['calibrated']:
                job['calibration_provenance'] = provenance
            if args.phase == 'six':
                job['command'] += ['--gate-rho', str(policy['selector_gate_rho'])]
            elif args.phase == 'gate-smoke':
                command = job['command']
                command[command.index('--gate-rho')+1] = str(policy['selector_gate_rho'])
    sources = [Path(__file__).resolve(), ROOT / "scripts/evaluate_pspr_holdout.py",
               ROOT / "scripts/calibrate_pspr_gate.py", ROOT / "scripts/measure_ranker_teacher_forced.py",
               ROOT / "scripts/dump_eval_traces.py", TAPS / "scripts/decode_lattice.py", TAPS / "model/utils.py",
               ROOT / "specforge/modeling/draft/pspr_slotdeep.py", ROOT / "specforge/modeling/draft/pspr_cloze.py"]
    for directory in [TAPS / 'models/Qwen3-4B'] + [export_path(a) / 'backbone' for a in RUNS]:
        for pattern in ('*.safetensors', '*.json', '*.py', '*.model', '*.txt'):
            sources += [p for p in directory.glob(pattern) if p.is_file()]
    if args.phase == 'gate-smoke':
        sources += [OUT / f'diagnostics_{a}_native{ext}' for a in RUNS for ext in ('.npz', '.json')]
    if args.phase == "confirmation":
        sources += [NEW / "manifest.json", NEW / "validation.jsonl", ARTIFACT]
    elif args.phase in {"six", "gate-smoke"}:
        sources += [prompts, ARTIFACT]
    else:
        sources += [OLD / "manifest.json", OLD / ("calibration.jsonl" if args.phase == "diagnostics" else "validation.jsonl")]
    hashes = {str(p): sha(p) for p in sources}

    def run(job):
        start = time.monotonic()
        provenance = dict(created_utc=datetime.now(timezone.utc).isoformat(), **job,
                          source_sha256=hashes, selector_sha256=sha(Path(job["export"]) / "selector.pt"),
                          scope="Frozen step7115 CE/KD; separately calibrated rho; target-greedy agreement only")
        write_json(OUT / (job["tag"] + "_PROVENANCE.json"), provenance)
        success = False
        try:
            command = job["command"]
            job_env = {**env, "CUDA_VISIBLE_DEVICES": job["gpu"]}
            logged(command, Path(job["cwd"]), job_env, OUT / (job["tag"] + ".log"))
            if args.phase == "smoke":
                stats_path = OUT / (job["tag"] + "_stats.json")
                stats_cmd = command[:-2] + ["--output", str(stats_path), "--gate-stats"]
                logged(stats_cmd, ROOT, job_env, OUT / (job["tag"] + "_stats.log"))
                a = json.loads(Path(job["result"]).read_text())
                b = json.loads(stats_path.read_text())
                old = json.loads(OLD_RESULTS[job["arm"]].read_text())
                expected = [r for r in old["results"] if r["prompt_index"] < 2]
                assert a["results"] == expected
                assert [{k: v for k, v in row.items() if k != "gate_stats"} for row in b["results"]] == expected
                assert a["skipped"] == b["skipped"] == [r for r in old["skipped"] if r["index"] < 2]
            elif args.phase == 'gate-smoke':
                import numpy as np
                from scripts.analyze_pspr_memory_s1 import risk_counts
                import math
                meta = json.loads(Path(job['result']).read_text())
                assert meta['gates'] and all(meta['gates'].values())
                assert 'G6 d-form gate == select_margin_gate' in meta['gates']
                with np.load(Path(job['result']).with_suffix('.npz'), allow_pickle=False) as new, np.load(
                    OUT / f"diagnostics_{job['arm']}_native.npz", allow_pickle=False) as old:
                    select = old['BASE_sample'] < 2
                    for key in ('kind', 'margin', 'slot', 'sample'):
                        assert np.array_equal(new['BASE_'+key], old['BASE_'+key][select]), key
                    threshold = math.log(meta['arguments']['gate_rho'])
                    counts = risk_counts(new['BASE_kind'], new['BASE_margin'], new['BASE_slot'], threshold)
                    c = meta['populations']['BASE']['counters']
                    assert counts['repaired'] == c['fix_recovered'] and counts['destroyed'] == c['base_right_destroyed']
            assert all(sha(p) == h for p, h in hashes.items()), "source/input changed during job"
            assert sha(Path(job["export"]) / "selector.pt") == provenance["selector_sha256"]
            success = True
            print("CALIBRATION_PHASE_JOB_DONE", job["tag"], flush=True)
        finally:
            write_json(OUT / (job["tag"] + "_EXIT.json"), dict(success=success, elapsed_seconds=time.monotonic()-start,
                       finished_utc=datetime.now(timezone.utc).isoformat()))
    with ThreadPoolExecutor(max_workers=len(jobs)) as executor:
        list(executor.map(run, jobs))


if __name__ == "__main__":
    main()
