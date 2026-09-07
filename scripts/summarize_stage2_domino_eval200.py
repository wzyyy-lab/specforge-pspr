"""Validate and summarize the fresh 1044-prompt, matched-H15 comparison."""
from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np

from run_stage2_domino_eval200 import (DATASETS, DOMINO, EXPECTED, MANIFEST, S2,
    SEED, TARGET, WORK, check_manifest)
from run_stage1_domino_h15_eval50 import digest, save_exclusive


def ratio(rows):
    a = sum(r["accepted_sum"] for r in rows)
    b = sum(r["num_blocks"] for r in rows)
    return dict(prompts=len(rows), accepted_sum=a, num_blocks=b, mean_acceptance=a / b)


def main():
    done = json.loads((WORK / "completion.json").read_text())
    assert done["status"] == "DONE" and done["input_hashes_unchanged"]
    manifest = check_manifest()
    metadata = {(r["dataset"], r["prompt_index"]): r for r in manifest["prompts"]}
    expected_jobs = {(method, s["dataset"], s["shard"]) for s in manifest["shards"] for method in ("stage2", "domino")}
    observed_jobs = set()
    modes = ("stage2_body", "stage2_full", "domino_h15")
    rows = {m: {} for m in modes}
    gate = defaultdict(Counter)
    artifacts = {}
    for record in done["results"]:
        assert record["exit_code"] == 0
        shard = record["shard"]
        job = (record["method"], shard["dataset"], shard["shard"])
        assert job in expected_jobs and job not in observed_jobs
        observed_jobs.add(job)
        path = Path(record["result"])
        artifacts[str(path)] = digest(path)
        assert artifacts[str(path)] == record["result_sha256"]
        data = json.loads(path.read_text())
        args = data["arguments"]
        assert data["proposal_slots"] == 15 and data["block_size"] == 16
        assert data["prompt_manifest_sha256"] == digest(shard["path"])
        assert args["max_samples"] == shard["count"] and args["shuffle_seed"] == SEED
        assert args["max_new_tokens"] == 256 and args["draft_attn"] == "sdpa"
        assert Path(args["target_model"]).resolve() == TARGET.resolve()
        assert not args["held_out"] and not args["eval_reserved"]
        mapping = {"oneshot": "stage2_body", "latgate": "stage2_full"} if record["method"] == "stage2" else {"domino_official": "domino_h15"}
        if record["method"] == "stage2":
            assert args["stop_policy"] == "official"
            assert (args["gate_rho"], args["gate_tau"], args["gate_theta"]) == (3, 0, 0)
            assert not args["gate_skip0"] and args["modes"] == "oneshot,latgate"
            assert Path(args["draft_model"]).resolve() == (S2 / "backbone").resolve()
            assert Path(args["lattice_head"]).resolve() == (S2 / "selector.pt").resolve()
            gate[shard["dataset"]].update(data["gate_stats"]["latgate"])
        else:
            assert data["stop_policy"] == "official" and args["max_proposals"] == 15 and not args["no_head"]
            assert Path(args["domino_draft"]).resolve() == DOMINO.resolve()
            assert data["proposal_cap"]["shift_label"] is True
            assert data["proposal_cap"]["backbone_input_block_size"] == 16
        local = json.loads(Path(shard["path"]).read_text())["prompts"]
        assert len(data["results"]) == shard["count"] * len(mapping)
        for row in data["results"]:
            mode = mapping[row["mode"]]
            p = local[row["prompt_index"]]
            key = (row["dataset"], p["canonical_prompt_index"])
            assert key not in rows[mode] and key in metadata
            assert row["prompt_sha256"] == p["prompt_sha256"] == metadata[key]["prompt_sha256"]
            al = row["acceptance_lengths"]
            assert al and all(type(x) is int and 1 <= x <= 16 for x in al)
            assert sum(al) == row["accepted_sum"] and len(al) == row["num_blocks"]
            assert abs(sum(al) / len(al) - row["mean_acceptance"]) < 1e-12
            rows[mode][key] = dict(row, mode=mode, prompt_index=key[1], local_prompt_index=row["prompt_index"],
                source_result=str(path))
    assert observed_jobs == expected_jobs and len(observed_jobs) == done["tasks_expected"] == 44
    assert all(set(v) == set(metadata) for v in rows.values())
    metrics, totals = {}, {}
    for m, all_rows in rows.items():
        metrics[m], totals[m] = {}, {}
        for ds in DATASETS:
            part = [all_rows[(ds, i)] for i in range(EXPECTED[ds])]
            totals[m][ds] = ratio(part)
            metrics[m][ds] = totals[m][ds]["mean_acceptance"]
        metrics[m]["macro"] = float(np.mean(list(metrics[m].values())))
        totals[m]["all"] = ratio(list(all_rows.values()))
        metrics[m]["micro"] = totals[m]["all"]["mean_acceptance"]
    rng = np.random.default_rng(SEED)
    reps = 10000
    macro_boot = {m: np.zeros(reps) for m in modes}
    intervals = {}
    for ds in DATASETS:
        n = EXPECTED[ds]
        ix = rng.integers(0, n, size=(reps, n))
        local_boot = {}
        for m in modes:
            arr = np.array([[rows[m][(ds, i)]["accepted_sum"], rows[m][(ds, i)]["num_blocks"]] for i in range(n)])
            sums = arr[ix].sum(axis=1)
            local_boot[m] = sums[:, 0] / sums[:, 1]
            macro_boot[m] += local_boot[m] / len(DATASETS)
        intervals[ds] = np.quantile(local_boot["stage2_full"] - local_boot["domino_h15"], [0.025, 0.975]).tolist()
    intervals["macro"] = np.quantile(macro_boot["stage2_full"] - macro_boot["domino_h15"], [0.025, 0.975]).tolist()
    categories = {}
    for cat in sorted({r["metadata"]["category"] for r in manifest["prompts"] if r["dataset"] == "mt-bench"}):
        keys = [key for key,r in metadata.items() if key[0] == "mt-bench" and r["metadata"]["category"] == cat]
        categories[cat] = {m: ratio([rows[m][k] for k in keys]) for m in modes}
    diagnostics = {}
    for ds, st in gate.items():
        assert st["blocks"] == totals["stage2_full"][ds]["num_blocks"]
        assert st["slots"] == 15 * st["blocks"]
        assert st["accepted_slots"] + st["blocks"] == totals["stage2_full"][ds]["accepted_sum"]
        diagnostics[ds] = dict(raw=dict(st),
            reachable_fixable_repair_rate=st["fix_recovered"] / st["fixable_n"],
            reachable_base_correct_destroy_rate=st["base_right_destroyed"] / st["base_right_n"],
            reachable_wrong_outside_top16_rate=st["unfixable_n"] / st["base_wrong_n"],
            reachable_fixable_true_best_but_gate_blocked_rate=st["fix_gate_blocked_true_best"] / st["fixable_n"])
    mix = json.loads((WORK / "TRAINING_MIX.json").read_text())
    exclude = {tuple(k) for entry in mix["exact_first_user_overlap_lower_bound"] for k in entry["benchmark"]}
    sensitivity = {}
    for m in modes:
        vals = {ds: ratio([row for key,row in rows[m].items() if key[0] == ds and key not in exclude]) for ds in DATASETS}
        sensitivity[m] = dict(per_domain=vals, macro=float(np.mean([v["mean_acceptance"] for v in vals.values()])))
    save_exclusive(WORK / "combined_results.json", dict(manifest_sha256=digest(MANIFEST),
        results=[r for m in modes for _,r in sorted(rows[m].items())]))
    save_exclusive(WORK / "comparison.json", dict(metrics=metrics, raw_totals=totals,
        manifest_sha256=digest(MANIFEST), result_sha256=artifacts, seed=SEED, counts=EXPECTED,
        bootstrap=dict(repetitions=reps, seed=SEED, paired_within_domain=True,
            caveat="Fixed checkpoints; prompt-composition sensitivity, not training-seed variance. HumanEval/MT-Bench are censuses; resampling is hypothetical prompt-mix sensitivity, not unobserved finite-test-set sampling error.",
            stage2_minus_domino_percentile95=intervals),
        mtbench_categories=categories, stage2_diagnostics=diagnostics,
        diagnosis_scope="Accepted reachable prefix + first rejected position; not a strict first-base-error-only repair rate and not a causal domain comparison.",
        known_exact_overlap_exclusion=dict(excluded_per_domain=dict(Counter(k[0] for k in exclude)), results=sensitivity,
            scope="Exclude only exact first-user matches to original 200k sample; not proof that remaining prompts are decontaminated.")))
    lines = ["# Domino H15 / Stage2：每域请求随机200条的扩样评测", "",
        f"seed={SEED}。GSM8K/MATH500/MBPP/Alpaca各200；HumanEval全164、MT-Bench全80，共1044条。",
        "三模式全部重新解码：Stage2@9052 backbone-only、Stage2完整selector、官方缓存Domino+实际H15候选限制。未重训/调参。", "",
        "## 平均接受长度", "",
        "| 数据集 | N | Domino H15 | Stage2仅骨干 | Stage2完整 | 完整−Domino | 相对Domino |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for ds in [*DATASETS, "macro", "micro"]:
        d,b,s = [metrics[m][ds] for m in ("domino_h15", "stage2_body", "stage2_full")]
        lines.append(f"| {ds} | {EXPECTED.get(ds,1044)} | {d:.6f} | {b:.6f} | {s:.6f} | {s-d:+.6f} | {(s/d-1)*100:+.3f}% |")
    lines += ["", "每域为Σacceptance_lengths/Σblocks，含+1。macro六域等权；micro按全部block汇总，不同域样本数并不相同。", "",
        "## 原始分子/分母", "", "| 数据集 | Domino接受数/block数 | Stage2骨干接受数/block数 | Stage2完整接受数/block数 |", "|---|---:|---:|---:|"]
    for ds in [*DATASETS, "all"]:
        values = [totals[m][ds] for m in ("domino_h15", "stage2_body", "stage2_full")]
        lines.append("| " + ds + " | " + " | ".join(f"{v['accepted_sum']}/{v['num_blocks']}" for v in values) + " |")
    lines += ["", "## 配对prompt bootstrap：Stage2完整−Domino", "",
        "10000次、固定seed，仅反映题目组成敏感性；不是训练seed方差。HumanEval/MT-Bench已测全量，其重采样区间不表示有限测试集有未测样本误差。", ""]
    for ds, (lo,hi) in intervals.items():
        lines.append(f"- {ds}: [{lo:+.6f}, {hi:+.6f}]")
    lines += ["", "## MT-Bench原始类别（第一轮）", "",
        "| 类别 | N | Domino | Stage2骨干 | Stage2完整 | 完整−Domino |", "|---|---:|---:|---:|---:|---:|"]
    for cat, vals in categories.items():
        d,b,s = [vals[m]["mean_acceptance"] for m in ("domino_h15", "stage2_body", "stage2_full")]
        lines.append(f"| {cat} | {vals['domino_h15']['prompts']} | {d:.6f} | {b:.6f} | {s:.6f} | {s-d:+.6f} |")
    lines += ["", "## Stage2可达位置诊断", "",
        "统计population是已接受前缀+首个拒绝位置，允许一个block修复多个原始错误；不是严格的每block首错修复率。各域错误难度/分布不同，不能据此因果归因。", "",
        "| 数据集 | 可达可修复错误修复率 | 可达原本正确token误改率 | 可达错误中真值不在top16 | 可修复中真值排第一但被门控拦截 |",
        "|---|---:|---:|---:|---:|"]
    for ds in DATASETS:
        d = diagnostics[ds]
        keys = ("reachable_fixable_repair_rate", "reachable_base_correct_destroy_rate", "reachable_wrong_outside_top16_rate", "reachable_fixable_true_best_but_gate_blocked_rate")
        lines.append("| " + ds + " | " + " | ".join(f"{d[k]:.2%}" for k in keys) + " |")
    lines += ["", "## 训练数据数量假说与重叠检查", "",
        "真实来源/regen保留/监督token供给见TRAINING_MIX.md和JSON。数量是供给统计，不是实际梯度质量或样本充分性的证明。",
        f"已知精确第一轮文本重叠排除数：{dict(Counter(k[0] for k in exclude))}。不是全量去污染检查。",
        f"保守排除这些原始训练抽样重叠后，Stage2宏平均{sensitivity['stage2_full']['macro']:.6f}，Domino宏平均{sensitivity['domino_h15']['macro']:.6f}。", "",
        "## 范围", "",
        "同manifest/target/非thinking/greedy/BF16/SDPA/256new tokens/committed-EOS/H15。权重与核心评测代码hash前后不变。",
        "原生Domino输入block16且shift_label=true，保留官方输入，只限制实际候选15；不是截断旧结果或改shift对齐。",
        "完整末block计入接受长度，可能超过EOS/实际输出预算，不等于有效吞吐。",
        "使用真实benchmark问题但以target模型greedy token定义接受，属于相对下游正确率的synthetic_proxy；未测答题准确率/速度/全量target-AR输出一致性。",
        "当前随机subset有历史重叠，并非全新未使用/去污染测试集。没有改变训练数据比例的对照，不能因果断言差距由数据数量导致或与数据无关。", ""]
    with (WORK / "COMPARISON.md").open("x") as out:
        out.write("\n".join(lines))
    print("EVAL200_SUMMARY_PASS")
    print("\n".join(lines[:20]))


if __name__ == "__main__":
    main()
