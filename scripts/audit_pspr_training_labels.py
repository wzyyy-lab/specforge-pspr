"""Read-only HF-greedy label consistency check on the actual training Arrow cache.

No regeneration, training, label replacement, or decoder changes are performed.
This measures same-tensor HF teacher-forced agreement, not exact speculative
decode replay or a causal attribution of acceptance failures.
"""
from __future__ import annotations

import argparse
import bisect
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def assistant_prefix_flags(ids, header, nonthinking):
    """Return whether each position follows an observed empty-think assistant header."""
    flags = []
    state = None
    for i in range(len(ids)):
        if ids[i:i+len(header)] == header:
            state = ids[i+len(header):i+len(header)+len(nonthinking)] == nonthinking
        flags.append(state)
    return flags


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-prefix", default="cache/eb66eb26294cf174cded4bc43dae1d2533b0d0cb727d4c4dd098791c0965f2a2")
    p.add_argument("--target-model", default="/kl_infra_infer_intern/wangzhuoyu/TAPS-SP/models/Qwen3-4B")
    p.add_argument("--samples", type=int, default=128)
    p.add_argument("--seed", type=int, default=20260905)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.samples < 1:
        p.error("samples must be positive")
    prefix = Path(args.cache_prefix)
    files = sorted(prefix.parent.glob(prefix.name + "_*.pkl"))
    if not files:
        raise FileNotFoundError(args.cache_prefix)
    from datasets import Dataset
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tables = [Dataset.from_file(str(path)) for path in files]
    ends = np.cumsum([len(d) for d in tables]).tolist()
    assert args.samples <= ends[-1]
    indices = sorted(np.random.default_rng(args.seed).choice(ends[-1], args.samples, replace=False).tolist())
    tok = AutoTokenizer.from_pretrained(args.target_model)
    target = AutoModelForCausalLM.from_pretrained(
        args.target_model, dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to("cuda").eval()
    header = tok.encode("<|im_start|>assistant\n", add_special_tokens=False)
    nonthinking = tok.encode("<think>\n\n</think>\n\n", add_special_tokens=False)
    special = set(tok.all_special_ids)
    results, totals = [], defaultdict(lambda: dict(tokens=0, mismatches=0, gap_ge_1_mismatches=0))
    for n, global_index in enumerate(indices):
        part = bisect.bisect_right(ends, global_index)
        local_index = global_index - (ends[part-1] if part else 0)
        row = tables[part][local_index]
        ids, mask = row["input_ids"][0], row["loss_mask"][0]
        assert len(ids) == len(mask) and len(ids) <= 3072
        flags = assistant_prefix_flags(ids, header, nonthinking)
        valid = np.flatnonzero(np.asarray(mask[1:]) > 0)
        record = dict(cache_file=str(files[part]), cache_row=local_index, global_cache_row=global_index,
                      input_ids_sha256=hashlib.sha256(np.asarray(ids, dtype=np.int64).tobytes()).hexdigest(),
                      strata={}, mismatch_examples=[])
        if len(valid) == 0:
            record["skipped"] = "no_supervised_next_token"
            results.append(record)
            continue
        tensor = torch.tensor(ids, device="cuda", dtype=torch.long)[None]
        positions = torch.tensor(valid, device="cuda", dtype=torch.long)
        with torch.inference_mode():
            logits = target(tensor, attention_mask=torch.ones_like(tensor), use_cache=False,
                            logits_to_keep=positions).logits[0]
            assert logits.shape[0] == len(valid)
            pred = logits.argmax(-1)
            corpus = tensor[0, positions + 1]
            top_value = logits.gather(-1, pred[:, None]).squeeze(-1).float()
            corpus_value = logits.gather(-1, corpus[:, None]).squeeze(-1).float()
            gap = (top_value-corpus_value).cpu().numpy()
            predicted, labels = pred.cpu().numpy(), corpus.cpu().numpy()
        counts = defaultdict(lambda: dict(tokens=0, mismatches=0, gap_ge_1_mismatches=0))
        for pos, expected, actual, g in zip(valid+1, labels, predicted, gap):
            wrong = bool(actual != expected)
            prefix_kind = ("with_empty_think" if flags[pos] else
                           "without_empty_think" if flags[pos] is False else "unidentified_prefix")
            strata = ["all", prefix_kind]
            if int(expected) not in special:
                strata += ["nonspecial", prefix_kind + "_nonspecial"]
            for kind in strata:
                for dst in (counts[kind], totals[kind]):
                    dst["tokens"] += 1
                    dst["mismatches"] += int(wrong)
                    dst["gap_ge_1_mismatches"] += int(wrong and g >= 1.)
            if wrong and len(record["mismatch_examples"]) < 20:
                record["mismatch_examples"].append(dict(position=int(pos), corpus_id=int(expected),
                    hf_greedy_id=int(actual), logit_gap=float(g), prefix_kind=prefix_kind))
        record["strata"] = dict(counts)
        results.append(record)
        del logits, tensor, positions, pred, corpus, top_value, corpus_value
        if (n+1) % 16 == 0:
            print("LABEL_AUDIT", n+1, "of", len(indices), flush=True)
    sources = [Path(__file__).resolve(), Path("specforge/data/parse.py"),
               Path("specforge/data/preprocessing.py"), Path("specforge/data/template.py")]
    sources += files
    sources += sorted(Path(args.target_model).glob("*.json"))
    sources += sorted(Path(args.target_model).glob("*.safetensors"))
    result = dict(arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        population_rows=ends[-1], sampled_rows=len(results), totals=dict(totals), per_sample=results,
        source_sha256={str(path.resolve()):sha(path) for path in sources},
        scope="Actual cached corpus next-token labels vs HF target argmax on the exact same input tensor, sdpa bf16 teacher forcing. Not tokenwise speculative replay; not proof of causal acceptance impact. Prefix strata refer to the visible serialized header, not an inferred original turn number.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as sink:
        json.dump(result,sink,indent=2)
        sink.write("\n")
    print("LABEL_AUDIT_DONE", json.dumps(result["totals"]), flush=True)


if __name__ == "__main__":
    main()

