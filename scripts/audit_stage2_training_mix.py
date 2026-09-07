"""Recover true PerfectBlend source labels and count Stage2 training input supply.

Read-only with respect to data, weights and training. Counts are corpus supply,
not realized optimizer gradient weights, and cannot establish a data-causal effect.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq

from run_stage2_domino_eval200 import MANIFEST, SF, WORK
from run_stage1_domino_h15_eval50 import digest, save_exclusive

RAW = SF / "cache/dataset/perfectblend_200k.jsonl"
REGEN = SF / "cache/dataset/perfectblend_qwen3-4b_regen.jsonl"
TURNS = SF / "cache/dataset/perfectblend_qwen3-4b_regen_turnwise_20260905.jsonl"
PARQUET = Path("/home/wangzhuoyu/sp-decoding-iclr/data/open-perfectblend/data")


def stream(path):
    h = hashlib.sha256()
    with path.open("rb") as reader:
        for line in reader:
            h.update(line)
            yield json.loads(line)
    file_hashes[str(path)] = h.hexdigest()


file_hashes = {}


def norm(text):
    return " ".join(text.split())


def main():
    sources = []
    for path in sorted(PARQUET.glob("train-*.parquet")):
        sources.append(pq.read_table(path, columns=["source"]).column("source").to_pylist())
        file_hashes[str(path)] = digest(path)
    assert len(sources) == 6
    def source(identifier):
        prefix, file_id, row_id = identifier.split("-")
        assert prefix == "pb"
        return sources[int(file_id)][int(row_id)]
    manifest = json.loads(MANIFEST.read_text())
    benchmark_keys = defaultdict(list)
    suffix = "\nPlease reason step by step, and put your final answer within \\boxed{}."
    for row in manifest["prompts"]:
        text = row["content"]
        if row["dataset"] in ("gsm8k", "math500"):
            assert text.endswith(suffix)
            text = text[:-len(suffix)]
        benchmark_keys[norm(text)].append((row["dataset"], row["prompt_index"]))
    stats = defaultdict(Counter)
    raw_ids, regen_ids = set(), set()
    overlap = []
    for row in stream(RAW):
        ident = row["id"]
        assert ident not in raw_ids
        raw_ids.add(ident)
        s = source(ident)
        stats[s]["sampled_documents"] += 1
        first = next((m["content"] for m in row["conversations"] if m["role"] == "user"), "")
        if norm(first) in benchmark_keys:
            overlap.append(dict(source_id=ident, source=s, benchmark=benchmark_keys[norm(first)]))
    assert len(raw_ids) == 200000
    print("TRAIN_MIX_RAW", {s:c["sampled_documents"] for s,c in stats.items()}, flush=True)
    for row in stream(REGEN):
        ident = row["id"]
        assert ident in raw_ids and ident not in regen_ids and row["status"] == "success"
        regen_ids.add(ident)
        s = source(ident)
        answers = sum(m["role"] == "assistant" for m in row["conversations"])
        stats[s]["regen_documents"] += 1
        stats[s]["regen_assistant_turns"] += answers
        stats[s]["regen_multiturn_documents"] += answers > 1
    assert len(regen_ids) == 199293
    usable_ids = defaultdict(set)
    turn_ids = set()
    for i, row in enumerate(stream(TURNS)):
        ident = row["source_id"]
        assert ident in regen_ids and row["id"] not in turn_ids
        turn_ids.add(row["id"])
        s = source(ident)
        usable_ids[s].add(ident)
        assert row["supervised_tokens"] == sum(row["loss_mask"])
        stats[s]["kept_training_turns"] += 1
        stats[s]["supervised_tokens"] += row["supervised_tokens"]
        stats[s]["input_tokens"] += len(row["input_ids"])
        stats[s]["truncated_turns"] += bool(row["truncated"])
        stats[s]["kept_first_answer_turns"] += row["assistant_turn"] == 1
        stats[s]["kept_later_answer_turns"] += row["assistant_turn"] > 1
        if (i + 1) % 50000 == 0:
            print("TRAIN_MIX_TURNS", i + 1, flush=True)
    for s in stats:
        stats[s]["usable_documents"] = len(usable_ids[s])
        stats[s]["skipped_answer_turns"] = stats[s]["regen_assistant_turns"] - stats[s]["kept_training_turns"]
    totals = Counter()
    for counts in stats.values():
        totals.update(counts)
    prior = json.loads(TURNS.with_suffix(".manifest.json").read_text())
    assert file_hashes[str(TURNS)] == prior["output_sha256"]
    for key, expected_key in [("kept_training_turns", "kept_turns"), ("supervised_tokens", "supervised_tokens"),
                              ("truncated_turns", "truncated_turns"), ("input_tokens", "input_tokens")]:
        assert totals[key] == prior["counts"][expected_key]
    explicit_math = ["meta-math/MetaMathQA", "HuggingFaceH4/orca-math-word-problems-200k"]
    math_counts = Counter()
    for s in explicit_math:
        math_counts.update(stats[s])
    math_share = {k: math_counts[k] / totals[k] for k in
                  ("sampled_documents", "regen_documents", "kept_training_turns", "supervised_tokens")}
    file_hashes[str(Path(__file__))] = digest(Path(__file__))
    save_exclusive(WORK / "TRAINING_MIX.json", dict(status="COUNTS_VERIFIED",
        stage2_input=str(TURNS), source_label_method="pb-file-row IDs joined to original parquet source column",
        sources={s:dict(c) for s,c in sorted(stats.items())}, totals=dict(totals),
        explicitly_math_sources=explicit_math, explicit_math_counts=dict(math_counts), explicit_math_share=math_share,
        exact_first_user_overlap_lower_bound=overlap,
        overlap_rule="Whitespace-normalized first-user equality; remove only evaluator-added fixed boxed-answer suffix for GSM8K/MATH500. Does not detect paraphrases/other templates or prove decontamination.",
        file_sha256=file_hashes,
        limitations=["Source names are not fine-grained semantic topic labels. Other sources also include math/code/chat.",
            "Supervised-token counts are eligible corpus tokens, not measured optimizer/anchor gradient mass.",
            "Counts alone cannot prove adequacy or establish that reweighting helps.",
            "Training is a nominal one-epoch run:9052x28=253456 examples versus253480 available turns; this is input-corpus composition, not exact per-step consumption."]))
    lines = ["# Stage2 训练语料来源核查", "",
        "从原始parquet的source字段按pb-{file}-{row}还原来源，贯通20万原始抽样、4B regen和实际Stage2 turn-wise输入。",
        "这是语料供给统计，不是根据回复关键词打标签，也不是实际optimizer梯度权重。", "",
        "| 原始来源 | 抽样文档 | regen文档 | 保留训练turn | turn占比 | 可监督token占比 | 截断turn | 丢弃turn |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for s,c in sorted(stats.items(), key=lambda x: -x[1]["kept_training_turns"]):
        lines.append(f"| {s} | {c['sampled_documents']} | {c['regen_documents']} | {c['kept_training_turns']} | {c['kept_training_turns']/totals['kept_training_turns']:.2%} | {c['supervised_tokens']/totals['supervised_tokens']:.2%} | {c['truncated_turns']} | {c['skipped_answer_turns']} |")
    lines += ["", f"合计：{totals['sampled_documents']}原始文档、{totals['regen_documents']} regen文档、{totals['kept_training_turns']}保留turn、{totals['supervised_tokens']}可监督token。", "",
        f"仅两个明确数学来源MetaMathQA+orca-math就占原始文档{math_share['sampled_documents']:.2%}、保留turn {math_share['kept_training_turns']:.2%}、可监督token {math_share['supervised_tokens']:.2%}。其他混合来源还可能含数学。",
        "这不支持直接断言数学数据整体很少，但也不能证明GSM8K细分题型、风格或错误修复场景覆盖充分。",
        "MT-Bench是多类别集合，且本轮仍只测第一轮，不能由低接受长度直接推断缺少多轮训练。", "",
        "来源数量只能检验明显的领域失衡，无法因果归因。要验证补数据是否有效，需要固定初始化/步数/损失，仅改变采样比例的对照；本次不启动训练。", "",
        "来源级精确数、输入hash、有限的第一轮文本精确重叠检查及局限见TRAINING_MIX.json。", ""]
    with (WORK / "TRAINING_MIX.md").open("x") as out:
        out.write("\n".join(lines))
    print("TRAIN_MIX_PASS", json.dumps(dict(totals)), "MATH_SHARES", json.dumps(math_share), flush=True)


if __name__ == "__main__":
    main()
