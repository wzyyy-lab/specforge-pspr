"""ID-resumable driver around SpecForge's existing, turn-wise regen function.

Every invocation appends to a NEW journal. An interrupted final journal line is
retained, ignored and retried by ID; previous files are never truncated. Final
success/error/skipped files are materialized in original input order. No old
4B file, global package, or original regen CLI default is changed.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time
import uuid

from scripts.regenerate_train_data import call_sglang, validate_regen_input, set_skipped
from scripts.validate_regenerated_data import validate_row


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def input_rows(path, limit=None):
    seen = set()
    with Path(path).open() as stream:
        for index, line in enumerate(stream):
            if limit is not None and index >= limit:
                break
            row = json.loads(line)
            identity = row.get("id")
            if not isinstance(identity, str) or not identity or identity in seen:
                raise ValueError(f"missing/duplicate/non-string source ID at line {index + 1}")
            seen.add(identity)
            yield row


def read_journals(directory, allowed_ids):
    results = {}
    for path in sorted(Path(directory).glob("journal-*.jsonl")):
        with path.open("rb") as stream:
            for number, line in enumerate(stream, 1):
                if not line.endswith(b"\n"):
                    # This is necessarily the last physical line; a new journal
                    # is used on restart, so the partial bytes remain untouched.
                    print(f"RETRY_PARTIAL_JOURNAL_TAIL {path}:{number}", flush=True)
                    continue
                row = json.loads(line)
                identity = row["id"]
                if identity not in allowed_ids or row["status"] not in ("success", "error", "skipped"):
                    raise ValueError(f"unexpected journal result at {path}:{number}")
                old = results.get(identity)
                if old and old["status"] == "success":
                    if row != old:
                        raise ValueError(f"conflicting completed ID {identity}")
                else:
                    results[identity] = row
    return results


def regenerate_one(args, address, row):
    invalid = validate_regen_input(row)
    if invalid:
        return set_skipped(copy.deepcopy(row), invalid)
    for attempt in range(args.attempts):
        try:
            result = call_sglang(args, address, copy.deepcopy(row))
            if result["status"] == "success":
                validate_row(result, expect_non_reasoning=True, strict_think_markers=True)
        except Exception as exc:
            result = dict(copy.deepcopy(row), status="error", error=f"{type(exc).__name__}: {exc}")
        if result["status"] != "error":
            return result
        if attempt + 1 < args.attempts:
            time.sleep(min(2 ** attempt, 8))
    return result


def materialize(args, results):
    paths = {"success": args.output_file_path,
             "error": args.output_file_path.with_name(args.output_file_path.stem + "_error.jsonl"),
             "skipped": args.output_file_path.with_name(args.output_file_path.stem + "_skipped.jsonl")}
    temps = {k: p.with_name(p.name + ".partial-" + uuid.uuid4().hex) for k, p in paths.items()}
    streams = {k: p.open("x") for k, p in temps.items()}
    counts, finishes = Counter(), Counter()
    try:
        for original in input_rows(args.input_file_path, args.num_samples):
            row = results[original["id"]]
            if row["status"] == "success":
                validate_row(row, expect_non_reasoning=True, strict_think_markers=True)
                # System/user turns must be byte-identical; original assistant
                # responses must not be substituted back into regenerated history.
                users = lambda r: [(m["role"], m["content"]) for m in r["conversations"]
                                   if m["role"] in ("system", "user")]
                if users(row) != users(original):
                    raise ValueError(f"regen changed a source prompt: {row['id']}")
                for message in row["conversations"]:
                    if message["role"] == "assistant":
                        finishes[message["generation"]["finish_reason"]] += 1
            counts[row["status"]] += 1
            streams[row["status"]].write(json.dumps(row, ensure_ascii=False) + "\n")
    finally:
        for stream in streams.values():
            stream.flush()
            os.fsync(stream.fileno())
            stream.close()
    if counts["success"] == 0:
        raise ValueError("no successful rows; journals and partial artifacts retained")
    for key, path in paths.items():
        if path.exists():
            if sha(path) != sha(temps[key]):
                raise FileExistsError(f"refusing to overwrite different existing data: {path}")
            temps[key].unlink()  # Only the identical temporary file made above.
        else:
            temps[key].rename(path)
    report = dict(counts=dict(counts), assistant_finish_reasons=dict(finishes),
        source_sha256=sha(args.input_file_path), model=str(args.model),
        protocol=dict(temperature=0.0, enable_thinking=False, max_tokens=args.max_tokens,
            multi_turn="every assistant regenerated; preceding regenerated answers form history",
            raw_token_ids_saved=False, resume="source ID, not completed line count"),
        outputs={key: dict(path=str(path), sha256=sha(path)) for key, path in paths.items()},
        status="complete" if not counts["error"] and not counts["skipped"] else "complete_with_exclusions")
    destination = args.output_file_path.with_suffix(".manifest.json")
    if destination.exists():
        if json.loads(destination.read_text()) != report:
            raise FileExistsError(f"different completed manifest: {destination}")
    else:
        with destination.open("x") as stream:
            json.dump(report, stream, indent=2)
            stream.write("\n")
    print("REGEN_COMPLETE", json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--server-address", nargs="+", required=True)
    parser.add_argument("--input-file-path", type=Path, required=True)
    parser.add_argument("--output-file-path", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=32, help="per server")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--num-samples", type=int)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--api-timeout", type=float, default=300.0)
    args = parser.parse_args()
    if min(args.concurrency, args.max_tokens, args.attempts, args.api_timeout) <= 0:
        parser.error("concurrency, limits and attempts must be positive")
    if args.num_samples is not None and args.num_samples <= 0:
        parser.error("num-samples must be positive")
    args.temperature, args.reasoning = 0.0, "disable"
    args.top_p = args.top_k = args.repetition_penalty = None
    args.is_gpt_oss, args.record_generation_metadata = False, True
    directory = args.output_file_path.with_suffix(".progress")
    directory.mkdir(parents=True, exist_ok=True)
    args.output_file_path.parent.mkdir(parents=True, exist_ok=True)
    with (directory / "lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        contract = dict(model=args.model, max_tokens=args.max_tokens, num_samples=args.num_samples,
            source_sha256=sha(args.input_file_path), model_config_sha256=sha(Path(args.model)/"config.json"),
            generation_code_sha256=sha(Path(__file__).with_name("regenerate_train_data.py")),
            driver_code_sha256=sha(__file__), temperature=0.0, enable_thinking=False)
        contract_path = directory / "contract.json"
        if contract_path.exists():
            if json.loads(contract_path.read_text()) != contract:
                raise ValueError("resume contract changed; use a different output, do not mix datasets")
        else:
            with contract_path.open("x") as stream:
                json.dump(contract, stream, indent=2)
        allowed_ids = {r["id"] for r in input_rows(args.input_file_path, args.num_samples)}
        results = read_journals(directory, allowed_ids)
        done_ids = {key for key, row in results.items() if row["status"] in ("success", "skipped")}
        final_manifest = args.output_file_path.with_suffix(".manifest.json")
        if final_manifest.exists():
            report = json.loads(final_manifest.read_text())
            for output in report["outputs"].values():
                if sha(output["path"]) != output["sha256"]:
                    raise ValueError("completed output checksum mismatch")
            print("REUSE_COMPLETE_REGEN", json.dumps(report["counts"]), flush=True)
            return
        print("REGEN_START", dict(total=len(allowed_ids), completed=len(done_ids),
            servers=args.server_address, concurrency_per_server=args.concurrency), flush=True)
        rows = (r for r in input_rows(args.input_file_path, args.num_samples) if r["id"] not in done_ids)
        journal = directory / (f"journal-{time.time_ns()}-{uuid.uuid4().hex[:8]}.jsonl")
        workers = args.concurrency * len(args.server_address)
        counts = Counter(results[key]["status"] for key in done_ids)
        # Each worker is replenished on ITS server, keeping per-server bounds.
        with journal.open("x") as sink, ThreadPoolExecutor(max_workers=workers) as pool:
            pending = {}
            for index in range(workers):
                row = next(rows, None)
                if row is None:
                    break
                address = args.server_address[index % len(args.server_address)]
                pending[pool.submit(regenerate_one, args, address, row)] = address
            while pending:
                finished, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in finished:
                    address = pending.pop(future)
                    result = future.result()
                    sink.write(json.dumps(result, ensure_ascii=False) + "\n")
                    sink.flush()
                    os.fsync(sink.fileno())
                    results[result["id"]] = result
                    counts[result["status"]] += 1
                    if sum(counts.values()) % 100 == 0:
                        print("REGEN_PROGRESS", json.dumps(dict(counts)), flush=True)
                    row = next(rows, None)
                    if row is not None:
                        pending[pool.submit(regenerate_one, args, address, row)] = address
        if set(results) != allowed_ids:
            raise ValueError("missing or extra source IDs at completion")
        materialize(args, results)


if __name__ == "__main__":
    main()
