"""Validate raw aligned samples and summarize the inference-only H15 comparison."""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_stage1_domino_h15_eval50 import DATASETS, OLD, PROMPTS, WORK, digest, save_exclusive


def main():
    completion = json.loads((WORK / "completion.json").read_text())
    assert completion["status"] == "DONE" and completion["input_and_code_hashes_unchanged"]
    provenance = json.loads((WORK / "launch_provenance.json").read_text())
    manifest = json.loads(PROMPTS.read_text())
    expected = {(p["dataset"], p["prompt_index"]): p["prompt_sha256"] for p in manifest["prompts"]}
    assert len(expected) == 300
    rows = defaultdict(dict)
    artifact_hashes = {}

    def consume(path, mappings, h):
        payload = json.loads(path.read_text())
        assert payload["proposal_slots"] == h
        assert payload["prompt_manifest_sha256"] == digest(PROMPTS)
        args = payload["arguments"]
        assert args["max_samples"] == 50 and args["max_new_tokens"] == 256
        assert args["shuffle_seed"] == 2026 and args["draft_attn"] == "sdpa"
        assert not args["eval_reserved"] and not args["held_out"]
        if "latgate" in mappings:
            assert args["stop_policy"] == "official"
            assert (args["gate_rho"], args["gate_tau"], args["gate_theta"]) == (3, 0, 0)
            assert not args["gate_skip0"]
        else:
            assert payload["stop_policy"] == "official" and not args["no_head"]
            if h == 15:
                assert args["max_proposals"] == 15
                assert payload["proposal_cap"]["shift_label"] is True
                assert payload["proposal_cap"]["backbone_input_block_size"] == 16
        for row in payload["results"]:
            if row["mode"] not in mappings:
                continue
            label = mappings[row["mode"]]
            key = (row["dataset"], row["prompt_index"])
            assert key in expected and row["prompt_sha256"] == expected[key]
            assert key not in rows[label]
            al = row["acceptance_lengths"]
            assert al and all(type(x) is int and 1 <= x <= h + 1 for x in al)
            assert sum(al) == row["accepted_sum"] and len(al) == row["num_blocks"]
            assert abs(sum(al) / len(al) - row["mean_acceptance"]) < 1e-12
            rows[label][key] = row
        artifact_hashes[str(path)] = digest(path)

    for ds in DATASETS:
        consume(WORK / f"stage1_t3_{ds}.json", {"oneshot": "s1_backbone", "latgate": "s1_t3"}, 15)
    for gpu in (6, 7):
        consume(WORK / f"domino_h15_gpu{gpu}.json", {"domino_official": "domino_h15"}, 15)
    consume(OLD / "slotdeep_stage2_step9052.json",
        {"oneshot": "s2_backbone", "latgate": "s2_step9052", "oracle16": "s2_oracle16"}, 15)
    consume(OLD / "domino_official.json", {"domino_official": "domino_native_h16"}, 16)
    for record in completion["results"]:
        assert record["result_sha256"] == artifact_hashes[str(WORK / (record["name"] + ".json"))]
    assert artifact_hashes[str(OLD / "slotdeep_stage2_step9052.json")] == provenance["hashes"][str(OLD / "slotdeep_stage2_step9052.json")]
    metrics, raw = {}, {}
    for label, samples in rows.items():
        assert set(samples) == set(expected)
        metrics[label], raw[label] = {}, {}
        for ds in DATASETS:
            selected = [samples[(ds, i)] for i in range(50)]
            accepted = sum(r["accepted_sum"] for r in selected)
            blocks = sum(r["num_blocks"] for r in selected)
            metrics[label][ds] = accepted / blocks
            raw[label][ds] = dict(prompts=50, accepted_sum=accepted, num_blocks=blocks)
        metrics[label]["macro"] = float(np.mean([metrics[label][ds] for ds in DATASETS]))
        total_a = sum(r["accepted_sum"] for r in samples.values())
        total_b = sum(r["num_blocks"] for r in samples.values())
        metrics[label]["micro"] = total_a / total_b
        raw[label]["all"] = dict(prompts=300, accepted_sum=total_a, num_blocks=total_b)

    rng = np.random.default_rng(2026)
    repeats = 10000
    bootstrap = {label: np.zeros(repeats) for label in ("s1_t3", "s2_step9052", "domino_h15")}
    for ds in DATASETS:
        indices = rng.integers(0, 50, size=(repeats, 50))
        for label in bootstrap:
            array = np.array([[rows[label][(ds, i)]["accepted_sum"],
                               rows[label][(ds, i)]["num_blocks"]] for i in range(50)])
            totals = array[indices].sum(axis=1)
            bootstrap[label] += totals[:, 0] / totals[:, 1] / len(DATASETS)
    intervals = {}
    for left, right in (("s1_t3", "domino_h15"), ("s2_step9052", "domino_h15"),
                        ("s2_step9052", "s1_t3")):
        intervals[f"{left}_minus_{right}"] = dict(
            point_difference=metrics[left]["macro"] - metrics[right]["macro"],
            percentile95=np.quantile(bootstrap[left] - bootstrap[right], [0.025, 0.975]).tolist())
    save_exclusive(WORK / "comparison.json", dict(metrics=metrics, raw_totals=raw,
        bootstrap=dict(seed=2026, repetitions=repeats,
            method="paired prompt resampling within each domain; equal-domain macro; fixed checkpoints",
            intervals=intervals), result_sha256=artifact_hashes,
        manifest_sha256=digest(PROMPTS), checks="300 aligned unique prompts per method; H/args/raw sums/exit hashes verified"))

    lines = ["# Stage1 / Stage2 / official Domino：同 H=15，每域50条", "",
        "Stage1 = T3（SlotDeep S1@7115 后 selector 续训1000步、官方 DFlash 冻结），是 Stage2 的实际起点；本轮没有重新选择最优checkpoint。",
        "Stage2 = 解冻后联合训练终点9052步。本次重新实跑 Stage1 和 Domino H15；Stage2复用同manifest的已完成结果。", "",
        "## 主结果：平均接受长度", "",
        "| 数据集 | 条数 | Stage1 T3 | Domino H15 | S1−Domino | S1相对Domino | Stage2@9052 | S2−Domino |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for ds in [*DATASETS, "macro", "micro"]:
        a, b, c = [metrics[label][ds] for label in ("s1_t3", "domino_h15", "s2_step9052")]
        label = {"macro": "六域等权宏平均", "micro": "全部block加权微平均"}.get(ds, ds)
        lines.append(f"| {label} | {50 if ds in DATASETS else 300} | {a:.6f} | {b:.6f} | {a-b:+.6f} | {(a/b-1)*100:+.3f}% | {c:.6f} | {c-b:+.6f} |")
    lines += ["", "每域均为 Σacceptance_lengths / Σ验证block数，不是先算每条均值再取平均。",
        "接受长度包含+1；H是实际proposal数15，单block上限16。Oracle@16中的16是每slot候选数K，概念不同。", "",
        "## 自身骨干与旧原生Domino参考", "",
        "| 数据集 | S1骨干-only | S1+selector | S2骨干-only | S2+selector | 旧Domino H16 | S2 oracle@16 |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for ds in [*DATASETS, "macro", "micro"]:
        numbers = [metrics[m][ds] for m in ("s1_backbone", "s1_t3", "s2_backbone", "s2_step9052", "domino_native_h16", "s2_oracle16")]
        lines.append("| " + ds + " | " + " | ".join(f"{v:.6f}" for v in numbers) + " |")
    lines += ["", "旧Domino H16只作原生设置参考，不混入同H15结论；oracle使用额外target信息，不是可部署速度结果。", "",
        "## 原始分子/分母", "", "| 数据集 | S1接受数/block数 | Domino H15接受数/block数 | S2接受数/block数 |",
        "|---|---:|---:|---:|"]
    for ds in [*DATASETS, "all"]:
        values = [raw[m][ds] for m in ("s1_t3", "domino_h15", "s2_step9052")]
        lines.append("| " + ds + " | " + " | ".join(f"{v['accepted_sum']}/{v['num_blocks']}" for v in values) + " |")
    lines += ["", "## 提示级配对bootstrap", "",
        "固定checkpoint，各域50条提示配对有放回重采样10000次，seed2026。区间反映提示抽样不确定性，不是训练seed方差。", ""]
    for name, result in intervals.items():
        lo, hi = result["percentile95"]
        lines.append(f"- {name}: macro差值 {result['point_difference']:+.6f}; percentile 95% CI [{lo:+.6f}, {hi:+.6f}]")
    lines += ["", "## H对齐和范围", "",
        "Domino原生block_size16且shift_label=true，原本实际预测16个；PSPR/SlotDeep实际预测15个。",
        "此次保留官方权重、配置、16位骨干输入和shift对齐；通过默认关闭的--max-proposals15，仅在官方方法的内存副本中限制k_draft。",
        "没有把旧结果截断、没有改官方模型文件、没有改Stage1/Stage2训练/架构/门控。",
        "真实GPU见证验证cap16与原版逐验证输入/接受长度/生成token一致；cap15保持骨干宽16、验证宽16(anchor+15)，首块前15候选一致。", "",
        "共同设置：同一300条manifest、每域50、seed2026、greedy/nonthinking、target BF16、SDPA、max_new_tokens256、官方committed-EOS停止。",
        "保持原始完整最后block计数，可能超过EOS/max_new_tokens后的有效输出数；这是原有接受长度口径，不是有效吞吐。",
        "这是已使用过的开发benchmark，历史记录明确有MATH500 index7重叠，不声称全新测试集或训练语料去重。",
        "本表只衡量相对target逐token一致的接受长度（synthetic_proxy，相对任务正确率而言）；不代表GSM8K答题正确率或HumanEval pass@1。",
        "没有延迟/速度、全量输出一致性、重复训练seed实验，不据此宣称更快、更准确或泛化已获证实。", "",
        "完整命令/模型及代码hash见launch_provenance.json；逐样本原始数组在各域JSON；退出码和末端hash检查见completion.json。", ""]
    with (WORK / "COMPARISON.md").open("x") as stream:
        stream.write("\n".join(lines))
    print("MATCHED_H15_SUMMARY_PASS")
    print("\n".join(lines[:23]))


if __name__ == "__main__":
    main()
