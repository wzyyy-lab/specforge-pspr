"""Turn-wise, current-answer-only tokenization of existing non-thinking regen data.

Does NOT regenerate answers or capture features. Training still obtains fresh
target hidden states through online SGLang. For each answer, reconstruct the
regen query: unchanged history texts, add_generation_prompt=True, thinking=False.
Only current answer text tokens are supervised. Do not invent EOS labels:
the original regen file retained neither stop-vs-length reasons nor token IDs.
Re-tokenization/backend numerical differences therefore remain possible.

Writes a new pre-tokenized JSONL supported by prepare_prompt_tasks. Original
data, generic parsers, model code and configuration defaults remain unchanged.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import itertools
import json
import multiprocessing as mp
import os
from pathlib import Path

TOKENIZER = None
OPTIONS = None


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def tokenize_turn(tokenizer, history, answer, *, max_length, min_answer_tokens=2):
    """Tokenize the generator prefix independently; never merge its last token."""
    prefix = tokenizer.apply_chat_template(history, tokenize=False,
        add_generation_prompt=True, enable_thinking=False)
    expected = "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    if not prefix.endswith(expected):
        raise ValueError("expected Qwen3 non-thinking generation prefix; refusing another template")
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    answer_ids = tokenizer.encode(answer, add_special_tokens=False)
    if set(answer_ids) & set(tokenizer.all_special_ids):
        return None, "special_in_answer"
    available = max(0, max_length - len(prefix_ids))
    kept = answer_ids[:available]
    if len(kept) < min_answer_tokens:
        return None, "insufficient_current_answer_after_right_truncation"
    return dict(input_ids=prefix_ids + kept, loss_mask=[0]*len(prefix_ids) + [1]*len(kept),
                prefix_tokens=len(prefix_ids), original_answer_tokens=len(answer_ids),
                supervised_tokens=len(kept), truncated=len(kept) < len(answer_ids)), None


def prepare_row(row, tokenizer, *, max_length, min_answer_tokens=2):
    if row.get("status") != "success":
        return [], Counter(source_rows=1, skipped_non_success=1)
    conversation = row["conversations"]
    if not conversation:
        raise ValueError("empty successful conversation")
    history, output = [], []
    counts = Counter(source_rows=1, successful_rows=1)
    turn = 0
    for index, message in enumerate(conversation):
        role, content = message["role"], message["content"]
        if not isinstance(content, str):
            raise ValueError("successful non-thinking regen content must be text")
        if role == "system":
            if index != 0:
                raise ValueError("system message is not first")
        elif role == "user":
            if history and history[-1]["role"] not in ("system", "assistant"):
                raise ValueError("nonalternating successful regen conversation")
        elif role == "assistant":
            if not history or history[-1]["role"] != "user":
                raise ValueError("assistant must follow its generation query")
            if not content.strip() or "<think>" in content or "</think>" in content:
                raise ValueError("not a successful non-thinking regenerated answer")
            turn += 1
            counts["assistant_turns"] += 1
            item, reason = tokenize_turn(tokenizer, history, content,
                max_length=max_length, min_answer_tokens=min_answer_tokens)
            if item is None:
                counts["skipped_" + reason] += 1
            else:
                item.update(id=f"{row['id']}::assistant:{turn}", source_id=row["id"],
                            assistant_turn=turn, source_message_index=index)
                output.append(item)
                counts["kept_turns"] += 1
                counts["supervised_tokens"] += item["supervised_tokens"]
                counts["input_tokens"] += len(item["input_ids"])
                counts["truncated_turns"] += int(item["truncated"])
        else:
            raise ValueError(f"unsupported successful regen role: {role!r}")
        # No artificial thinking markup in history: regen saves plain response
        # text and re-renders that history on the next API call.
        history.append(dict(role=role, content=content))
    if history[-1]["role"] != "assistant":
        raise ValueError("successful regen conversation must end in an answer")
    return output, counts


def init_worker(target_model, max_length, min_answer_tokens):
    global TOKENIZER, OPTIONS
    from transformers import AutoTokenizer
    TOKENIZER = AutoTokenizer.from_pretrained(target_model)
    OPTIONS = dict(max_length=max_length, min_answer_tokens=min_answer_tokens)


def process_row(row):
    try:
        return prepare_row(row, TOKENIZER, **OPTIONS)
    except Exception as exc:
        raise ValueError(f"source_id={row.get('id')!r}: {exc}") from exc


def input_rows(path, limit):
    seen = set()
    with path.open() as stream:
        for line in itertools.islice(stream, limit or None):
            row = json.loads(line)
            identity = row["id"]
            if identity in seen:
                raise ValueError(f"duplicate source id: {identity!r}")
            seen.add(identity)
            yield row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=3072)
    parser.add_argument("--min-answer-tokens", type=int, default=2)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--max-source-rows", type=int, default=0)
    args = parser.parse_args()
    if min(args.max_length, args.min_answer_tokens, args.workers) < 1 or args.max_source_rows < 0:
        parser.error("length/minimum/workers must be positive; max-source-rows must be nonnegative")
    manifest_path = args.output.with_suffix(".manifest.json")
    if args.output.exists() or manifest_path.exists():
        raise FileExistsError("refusing to overwrite tokenized data or provenance")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    hashes = {str(args.source.resolve()): sha(args.source), str(Path(__file__).resolve()): sha(__file__)}
    for path in sorted(args.target_model.glob("*.json")):
        hashes[str(path.resolve())] = sha(path)
    counts = Counter()
    # imap preserves file/turn order irrespective of worker completion order.
    with mp.get_context("spawn").Pool(args.workers, initializer=init_worker,
            initargs=(str(args.target_model), args.max_length, args.min_answer_tokens)) as pool:
        with args.output.open("x") as sink:
            for rows, batch_counts in pool.imap(process_row,
                    input_rows(args.source, args.max_source_rows), chunksize=32):
                counts.update(batch_counts)
                for row in rows:
                    sink.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                if counts["source_rows"] % 10000 == 0:
                    print("TURN_DATA", dict(counts), flush=True)
    if not counts["kept_turns"]:
        raise ValueError("no trainable turns; incomplete output retained for diagnosis")
    result = dict(created_utc=datetime.now(timezone.utc).isoformat(), scope=__doc__,
        arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        counts=dict(counts), source_sha256=hashes, output_sha256=sha(args.output),
        package_versions={k:importlib.metadata.version(k) for k in ("transformers","tokenizers")},
        eos_policy="No fabricated EOS/EOT/newline targets; original regen stop reason was not retained.",
        ordering="source file order, then assistant turn order; online trainer shuffles prompts separately",
        limitations="Turn-wise sampling and content-only masks also change the training population; not a prefix-only training ablation. Saved answer text is re-tokenized, not original generation token IDs.")
    with manifest_path.open("x") as sink:
        json.dump(result, sink, indent=2)
        sink.write("\n")
    print("TURN_DATA_DONE", json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
