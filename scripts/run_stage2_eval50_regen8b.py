"""Durable dependency queue: existing Stage2 -> 6x50 comparison -> 8B PB regen.

Never starts/restarts training and never kills existing jobs. The already
running shell has parsed its historical 6x20 evaluation, so that auxiliary run
is allowed to finish before this independent 6x50 protocol begins.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shlex
import signal
import socket
import subprocess
import sys
import time
import urllib.request

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO.parent
TAPS = ROOT / "TAPS-SP"
WORK = REPO / "outputs/STAGE2_EVAL50_REGEN8B_20260906"
MODEL8 = Path("/kl_infra_infer_intern/models/Qwen3-8B")
MODEL4 = TAPS / "models/Qwen3-4B"
DOMINO = TAPS / "models/Qwen3-4B-Domino-b16"
OVERLAY = "/home/wangzhuoyu/sglang-patched-0.5.15"
SESSION = "slotdeep_stage2_20260906"
RUN = "qwen3-4b-pspr-slotdeep-stage2-20260906"
CHECKPOINT = REPO / f"outputs/{RUN}/{RUN}-step9052/training_state.pt"
EXPORT = REPO / f"outputs/{RUN}_step9052_decode"
SOURCE = REPO / "cache/dataset/perfectblend_200k.jsonl"
REGEN = REPO / "cache/dataset/perfectblend_qwen3-8b_regen_20260906.jsonl"
PROMPTS = WORK / "prompts50.json"
DATASETS = ("gsm8k", "math500", "humaneval", "mbpp", "alpaca", "mt-bench")
SLOT_RESULT = WORK / "slotdeep_stage2_step9052.json"
DOMINO_RESULT = WORK / "domino_official.json"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def env(gpu=None, sglang=False):
    result = dict(os.environ)
    result.update(PYTHONPATH=f"{OVERLAY}:." if sglang else ".",
        OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4",
        TOKENIZERS_PARALLELISM="false", SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK="1",
        HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1",
        no_proxy="localhost,127.0.0.1", NO_PROXY="localhost,127.0.0.1")
    if gpu is not None:
        result["CUDA_VISIBLE_DEVICES"] = str(gpu)
    return result


def status(phase, **details):
    payload = dict(utc=datetime.now(timezone.utc).isoformat(), phase=phase, **details)
    WORK.mkdir(parents=True, exist_ok=True)
    temporary = WORK / "status.json.tmp"
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(WORK / "status.json")
    with (WORK / "events.jsonl").open("a") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False) + "\n")
    print("QUEUE_STATUS", json.dumps(payload, ensure_ascii=False), flush=True)


def run_command(command, log, cwd=REPO, gpu=None, sglang=False):
    print("RUN", f"gpu={gpu}", shlex.join(map(str, command)), flush=True)
    with Path(log).open("a") as stream:
        stream.write(f"\nINVOCATION {datetime.now(timezone.utc).isoformat()} {shlex.join(map(str, command))}\n")
        stream.flush()
        subprocess.run(list(map(str, command)), cwd=cwd, env=env(gpu, sglang),
            stdout=stream, stderr=subprocess.STDOUT, check=True)


def gpu_memory():
    raw = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used",
        "--format=csv,noheader,nounits"], text=True)
    return {int(row.split(",")[0]): int(row.split(",")[1]) for row in raw.splitlines()}


def wait_free(gpus):
    consecutive = 0
    while consecutive < 3:
        memory = gpu_memory()
        if all(memory[gpu] < 500 for gpu in gpus):
            consecutive += 1
        else:
            consecutive = 0
            print("WAIT_FREE_GPUS", memory, flush=True)
        if consecutive < 3:
            time.sleep(15)


def pane_state():
    result = subprocess.run(["tmux", "list-panes", "-t", SESSION,
        "-F", "#{pane_dead}|#{pane_dead_status}|#{pane_pid}"], capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError("Stage2 dependency session missing; refusing to infer completion")
    rows = result.stdout.strip().splitlines()
    if len(rows) != 1:
        raise RuntimeError("unexpected Stage2 pane layout")
    dead, code, pid = rows[0].split("|")
    return dict(dead=dead == "1", exit_code=int(code) if code else None, pid=int(pid))


def preflight():
    from safetensors import safe_open
    from transformers import AutoTokenizer
    from scripts.regenerate_by_id import input_rows
    WORK.mkdir(parents=True, exist_ok=True)
    config = json.loads((MODEL8 / "config.json").read_text())
    if (config["hidden_size"], config["num_hidden_layers"], config["vocab_size"]) != (4096, 36, 151936):
        raise ValueError("not the expected Qwen3-8B architecture")
    index = json.loads((MODEL8 / "model.safetensors.index.json").read_text())
    seen, shards, parameters = {}, {}, 0
    for name in sorted(set(index["weight_map"].values())):
        path = MODEL8 / name
        with safe_open(path, framework="pt", device="cpu") as stream:
            keys = list(stream.keys())
            shards[name] = dict(bytes=path.stat().st_size, tensors=len(keys))
            for key in keys:
                if key in seen:
                    raise ValueError(f"duplicate tensor: {key}")
                tensor = stream.get_slice(key)
                if tensor.get_dtype() != "BF16":
                    raise ValueError("expected unquantized BF16 target")
                parameters += math.prod(tensor.get_shape())
                seen[key] = name
    if seen != index["weight_map"] or parameters != 8190735360:
        raise ValueError("Qwen3-8B shards/index mismatch")
    tokenizer = AutoTokenizer.from_pretrained(MODEL8, local_files_only=True)
    witness = tokenizer.apply_chat_template([dict(role="user", content="What is 1+1?")],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    if not witness.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n"):
        raise ValueError("unexpected Qwen3 non-thinking template")
    source_rows = sum(1 for _ in input_rows(SOURCE))
    if source_rows != 200000:
        raise ValueError(f"expected 200000 original PerfectBlend rows, got {source_rows}")
    run_command(["python", "-u", "scripts/eval_protocol.py", "--output", PROMPTS,
        "--max-samples", "50", "--shuffle-seed", "2026"], WORK / "prepare_prompts.log", cwd=TAPS)
    verification = subprocess.check_output(["python", "-c",
        'import json,sglang,sgl_kernel,importlib.metadata as m; '
        'assert sglang.__version__=="0.5.15"; '
        'assert sgl_kernel.__version__=="0.4.4"; '
        'assert m.version("sglang-kernel")=="0.4.4+cu129"; '
        'assert m.version("torch")=="2.11.0+cu129"; '
        'assert m.version("transformers")=="5.12.1"; '
        'print(json.dumps({"sglang":sglang.__version__,"sglang_file":sglang.__file__, '
        '"kernel":m.version("sglang-kernel"),"torch":m.version("torch"), '
        '"transformers":m.version("transformers")}))'], cwd=REPO, env=env(sglang=True), text=True)
    # Parse the exact future server flags without constructing a model/server.
    run_command(["python", "-c",
        'import argparse; from sglang.srt.server_args import ServerArgs; '
        'from scripts.run_stage2_eval50_regen8b import server_command; '
        'p=argparse.ArgumentParser(); ServerArgs.add_cli_args(p); '
        'a=p.parse_args(server_command(0)[4:]); '
        'assert a.model_path=="/kl_infra_infer_intern/models/Qwen3-8B"; '
        'assert a.port==32100; print("EXACT_SERVER_FLAGS_PARSE_PASS")'],
        WORK / "server_cli_preflight.log", sglang=True)
    report = dict(model_path=str(MODEL8), model_shards=shards, parameter_count=parameters,
        weights_header_and_index_valid=True, gpu_generation_witness="pending until GPUs released",
        model_config_sha256=sha(MODEL8 / "config.json"),
        model_index_sha256=sha(MODEL8 / "model.safetensors.index.json"),
        tokenizer_config_sha256=sha(MODEL8 / "tokenizer_config.json"),
        source=str(SOURCE), source_rows=source_rows, source_sha256=sha(SOURCE),
        prompts_sha256=sha(PROMPTS), packages=json.loads(verification.strip().splitlines()[-1]),
        source_code_sha256={str(p): sha(p) for p in [Path(__file__),
            REPO / "scripts/regenerate_by_id.py", REPO / "scripts/regenerate_train_data.py",
            TAPS / "scripts/decode_lattice.py", TAPS / "scripts/bench_official_domino.py",
            TAPS / "scripts/eval_protocol.py", DOMINO / "dflash.py", TAPS / "model/dflash.py",
            REPO / "specforge/modeling/draft/pspr_slotdeep.py"]},
        dependency=pane_state(), gpu_memory_mib=gpu_memory(),
        evaluation=dict(samples_per_domain=50, domains=DATASETS, greedy=True,
            max_new_tokens=256, thinking=False, seed=2026, slotdeep_gate_rho=3,
            stop_policy="official committed-EOS; pending bonus not examined",
            slotdeep_proposal_slots=15, domino_proposal_slots=16,
            primary_metric="sum acceptance_lengths / number of blocks; includes +1; macro over six domains",
            caveat="expanded previously-used benchmark, not a newly untouched heldout test"),
        regen=dict(gpus=list(range(8)), concurrency_per_server=32, max_tokens=2048,
            temperature=0, enable_thinking=False, model_context_length=40960,
            output=str(REGEN), start="after successful 50-prompt comparison"))
    (WORK / "preflight.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print("PREFLIGHT_PASS_CPU_ONLY", json.dumps(report), flush=True)


def verify_final():
    import torch
    if not CHECKPOINT.is_file() or not (EXPORT / "backbone/model.safetensors").is_file():
        raise FileNotFoundError("final checkpoint/backbone missing")
    payload = torch.load(EXPORT / "selector.pt", map_location="cpu", weights_only=False)
    if payload["global_step"] != 9052 or payload["model_type"] != "SlotDeepCorrector":
        raise ValueError("wrong exported endpoint")
    if payload["training_policy"]["backbone"] != "trained":
        raise ValueError("export does not record a jointly trained backbone")
    for tensor in payload["state_dict"].values():
        if not torch.isfinite(tensor).all():
            raise ValueError("nonfinite selector tensor")
    export_log = REPO / "outputs/EXPORT_SLOTDEEP_STAGE2_STEP9052_20260906.log"
    if "max|delta|=0" not in export_log.read_text():
        raise ValueError("export roundtrip witness not found")
    report = dict(checkpoint=str(CHECKPOINT), export=str(EXPORT), global_step=9052,
        backbone_sha256=sha(EXPORT / "backbone/model.safetensors"),
        selector_sha256=sha(EXPORT / "selector.pt"),
        domino_sha256=sha(DOMINO / "model.safetensors"), target_config_sha256=sha(MODEL4 / "config.json"))
    (WORK / "final_artifacts.json").write_text(json.dumps(report, indent=2) + "\n")


def verify_unchanged_preflight():
    report = json.loads((WORK / "preflight.json").read_text())
    expected = {**report["source_code_sha256"], str(SOURCE): report["source_sha256"],
        str(PROMPTS): report["prompts_sha256"], str(MODEL8 / "config.json"): report["model_config_sha256"],
        str(MODEL8 / "model.safetensors.index.json"): report["model_index_sha256"]}
    changed = [path for path, digest in expected.items() if sha(path) != digest]
    if changed:
        raise RuntimeError(f"queued inputs/code changed since preflight: {changed}")


def summarize():
    slot, domino = json.loads(SLOT_RESULT.read_text()), json.loads(DOMINO_RESULT.read_text())
    expected_hash = sha(PROMPTS)
    for payload in (slot, domino):
        if payload["prompt_manifest_sha256"] != expected_hash:
            raise ValueError("prompt manifest hash mismatch")
        if payload["arguments"]["max_new_tokens"] != 256:
            raise ValueError("token budget mismatch")
    if slot["arguments"]["stop_policy"] != "official" or domino["stop_policy"] != "official":
        raise ValueError("EOS policy mismatch")
    prompts = json.loads(PROMPTS.read_text())["prompts"]
    metrics, raw_totals = {}, {}
    for mode, payload in (("latgate", slot), ("oneshot", slot), ("oracle16", slot), ("domino_official", domino)):
        metrics[mode], raw_totals[mode] = {}, {}
        for ds in DATASETS:
            rows = [row for row in payload["results"] if row["dataset"] == ds and row["mode"] == mode]
            expected = {(r["prompt_index"], r["prompt_sha256"]) for r in prompts if r["dataset"] == ds}
            if len(rows) != 50 or {(r["prompt_index"], r["prompt_sha256"]) for r in rows} != expected:
                raise ValueError(f"{mode}/{ds}: missing/duplicate/mismatched prompts")
            limit = int(payload["proposal_slots"]) + 1
            for row in rows:
                al = row["acceptance_lengths"]
                if not al or not all(isinstance(x, int) and 1 <= x <= limit for x in al):
                    raise ValueError("invalid raw acceptance lengths")
                if sum(al) != row["accepted_sum"] or len(al) != row["num_blocks"]:
                    raise ValueError("raw acceptance counters disagree")
            total = sum(r["accepted_sum"] for r in rows)
            blocks = sum(r["num_blocks"] for r in rows)
            metrics[mode][ds] = total / blocks
            raw_totals[mode][ds] = dict(accepted_sum=total, num_blocks=blocks, prompts=50)
        metrics[mode]["macro"] = sum(metrics[mode][ds] for ds in DATASETS)/len(DATASETS)
        metrics[mode]["micro"] = (sum(raw_totals[mode][ds]["accepted_sum"] for ds in DATASETS) /
                                   sum(raw_totals[mode][ds]["num_blocks"] for ds in DATASETS))
    summary = dict(metrics=metrics, raw_totals=raw_totals, gate_stats=slot.get("gate_stats"),
        prompts_sha256=expected_hash, caveats=[
            "Native block conventions: SlotDeep 15 proposals (+1), official Domino 16 proposals (+1).",
            "Oracle16 uses privileged target calls and is not a deployable speed baseline.",
            "First turn only; same greedy BF16 target, SDPA, 256-token budget and committed-EOS rule.",
            "This expanded benchmark is not untouched; no statistical significance or latency claim."])
    (WORK / "comparison.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    lines = ["# SlotDeep Stage2 与官方 Domino：六域各 50 条", "",
        "固定同一 300 条输入，greedy / 非 thinking / max_new_tokens=256 / seed=2026。",
        "接受长度按每域总接受数÷总 block 数计算；macro 为六域等权平均。", "",
        "| 数据集 | Stage2 backbone only | SlotDeep Stage2 | 官方 Domino | Stage2 Oracle@16 | SlotDeep−Domino |",
        "|---|---:|---:|---:|---:|---:|"]
    for ds in (*DATASETS, "macro", "micro"):
        vals = [metrics[m][ds] for m in ("oneshot", "latgate", "domino_official", "oracle16")]
        lines.append(f"| {ds} | " + " | ".join(f"{v:.4f}" for v in vals) + f" | {vals[1]-vals[2]:+.4f} |")
    lines += ["", "注意：保留官方原生设置，Domino 是 16 个 proposal，SlotDeep 是 15 个；",
        "两者报告都含 +1，且都只在已提交 EOS 时停止，不把待提交 bonus 的 EOS 提前判停。",
        "Oracle@16 是额外访问 target 得到的参考上界，不能当成部署速度。",
        "这是扩展后的既有 benchmark，不声称为未使用过的测试集，也不据单次点估计宣称显著提升。",
        "逐样本原始值、输入 hash、门控统计见同目录 JSON。本报告不修改任何训练方案。", ""]
    (WORK / "COMPARISON.md").write_text("\n".join(lines))


def evaluate():
    common = ["--target-model", MODEL4, "--datasets", ",".join(DATASETS), "--max-samples", "50",
        "--max-new-tokens", "256", "--shuffle-seed", "2026", "--prompt-manifest", PROMPTS]
    slot_cmd = ["python", "-u", "scripts/decode_lattice.py", *common,
        "--draft-model", EXPORT / "backbone", "--lattice-head", EXPORT / "selector.pt",
        "--modes", "oneshot,latgate,oracle16", "--gate-tau", "0", "--gate-rho", "3",
        "--gate-theta", "0", "--gate-stats", "--draft-attn", "sdpa", "--stop-policy", "official",
        "--json-output", SLOT_RESULT]
    domino_cmd = ["python", "-u", "scripts/bench_official_domino.py", *common,
        "--domino-draft", DOMINO, "--draft-attn", "sdpa", "--json-output", DOMINO_RESULT]
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = []
        for command, output, gpu in ((slot_cmd, SLOT_RESULT, 7), (domino_cmd, DOMINO_RESULT, 6)):
            if not output.exists():
                futures.append(pool.submit(run_command, command, output.with_suffix(".log"), TAPS, gpu))
        for future in futures:
            future.result()
    summarize()


def server_command(gpu):
    return ["python", "-u", "-m", "sglang.launch_server", "--model-path", str(MODEL8),
        "--mem-fraction-static", "0.80", "--tp", "1", "--cuda-graph-max-bs", "64",
        "--host", "127.0.0.1", "--port", str(32100 + gpu * 10), "--dtype", "bfloat16",
        "--random-seed", "42", "--context-length", "40960"]


def wait_server(process, address):
    deadline = time.monotonic() + 900
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"8B server {address} exited {process.returncode}")
        try:
            with opener.open(f"http://{address}/health", timeout=5) as response:
                if response.status == 200:
                    return
        except (OSError, TimeoutError):
            pass
        time.sleep(5)
    raise TimeoutError(f"8B server {address} did not become healthy within 15 min")


def regenerate():
    from types import SimpleNamespace
    from scripts.regenerate_train_data import call_sglang
    from scripts.validate_regenerated_data import validate_row
    processes, logs, addresses = [], [], []
    try:
        status("REGEN8B_SERVER_SMOKE", gpus=list(range(8)))
        for gpu in range(8):
            port = 32100 + gpu * 10
            with socket.socket() as probe:
                probe.bind(("127.0.0.1", port))
            log = (WORK / f"sglang_qwen3_8b_gpu{gpu}.log").open("a")
            logs.append(log)
            command = server_command(gpu)
            log.write("\nINVOCATION " + shlex.join(command) + "\n")
            log.flush()
            process = subprocess.Popen(command, cwd=REPO, env=env(gpu, sglang=True),
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            processes.append(process)
            addresses.append(f"127.0.0.1:{port}")
        args = SimpleNamespace(model=str(MODEL8), temperature=0.0, reasoning="disable",
            max_tokens=64, top_p=None, top_k=None, repetition_penalty=None, is_gpt_oss=False,
            record_generation_metadata=True, api_timeout=180.0)
        witnesses = []
        for process, address in zip(processes, addresses):
            wait_server(process, address)
            row = dict(id="8b-two-turn-witness", conversations=[
                dict(role="user", content="What is 1+1? Answer with only the number."),
                dict(role="assistant", content="OLD_ANSWER_MUST_NOT_ENTER_NEXT_QUERY"),
                dict(role="user", content="Add 3 to your previous answer. Answer with only the number.")])
            output = call_sglang(args, address, row)
            validate_row(output, expect_non_reasoning=True, strict_think_markers=True)
            answers = [m["content"].strip() for m in output["conversations"] if m["role"] == "assistant"]
            if answers != ["2", "5"]:
                raise ValueError(f"8B real two-turn witness failed at {address}: {answers!r}")
            witnesses.append(dict(address=address, result=output))
        (WORK / "qwen3_8b_gpu_witnesses.json").write_text(json.dumps(witnesses, indent=2) + "\n")
        smoke = WORK / "perfectblend_8b_smoke32.jsonl"
        base = ["python", "-u", "scripts/regenerate_by_id.py", "--model", MODEL8,
            "--server-address", *addresses, "--input-file-path", SOURCE, "--max-tokens", "2048"]
        run_command([*base, "--concurrency", "2", "--num-samples", "32",
            "--output-file-path", smoke], WORK / "regen_smoke32.log")
        smoke_summary = json.loads(smoke.with_suffix(".manifest.json").read_text())
        if smoke_summary["counts"].get("error", 0) or smoke_summary["counts"].get("success", 0) < 30:
            raise ValueError("PerfectBlend 8B smoke did not meet >=30/32 success and zero-error gate")
        status("REGEN8B_RUNNING", gpus=list(range(8)), source=str(SOURCE), output=str(REGEN),
            smoke=smoke_summary["counts"], servers=addresses)
        run_command([*base, "--concurrency", "32", "--output-file-path", REGEN], WORK / "regen_full.log")
        run_command(["python", "scripts/validate_regenerated_data.py", "--data-path", REGEN,
            "--expect-non-reasoning", "--strict-think-markers"], WORK / "regen_validation.log")
    finally:
        # Only process groups created by this function; never broad pkill or GPU reset.
        for process in processes:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        for process in processes:
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)
        for log in logs:
            log.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true", help="CPU-only verification; no GPU allocations")
    args = parser.parse_args()
    if args.preflight:
        preflight()
        return
    WORK.mkdir(parents=True, exist_ok=True)
    with (WORK / "queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            preflight()
            status("WAIT_STAGE2_AND_LEGACY_EVAL", dependency_session=SESSION,
                expected_final_step=9052, note="do not alter the live parsed 20-prompt auxiliary pipeline")
            while True:
                state = pane_state()
                if state["dead"]:
                    if state["exit_code"] != 0:
                        raise RuntimeError(f"Stage2 dependency failed: {state}")
                    break
                time.sleep(30)
            status("VERIFY_FINAL_ARTIFACTS")
            verify_unchanged_preflight()
            verify_final()
            wait_free(list(range(8)))
            status("EVAL50_RUNNING", slotdeep_gpu=7, domino_gpu=6, samples_per_domain=50, domains=DATASETS)
            evaluate()
            status("EVAL50_COMPLETE", comparison=str(WORK / "COMPARISON.md"))
            wait_free(list(range(8)))
            verify_unchanged_preflight()
            regenerate()
            report = json.loads(REGEN.with_suffix(".manifest.json").read_text())
            status("DONE", comparison=str(WORK / "COMPARISON.md"), regen=str(REGEN), regen_counts=report["counts"])
        except BaseException as exc:
            status("FAILED", error=f"{type(exc).__name__}: {exc}")
            raise


if __name__ == "__main__":
    main()
