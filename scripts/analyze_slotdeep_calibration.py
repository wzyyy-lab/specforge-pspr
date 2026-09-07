"""Fixed PB confirmation and historical six-domain calibration comparisons.

No fitting, sweep or checkpoint selection. Reuses the audited aggregation and
prompt bootstrap functions; calibrated rho is the sole serving-policy exception.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.analyze_pspr_holdout import compare
from scripts.analyze_pspr_memory_s1 import DOMAINS, SEMANTICS, load, metrics, bootstrap_delta, diagnostics
from scripts.calibrate_pspr_gate import METHOD, sha
from scripts.run_slotdeep_calibration import ROOT, TAPS, OUT, ARTIFACT, RUNS, export_path, write_json

NATIVE_SIX = dict(CE=ROOT / 'outputs/EVAL_SLOTDEEP_TURN_S1_STEP7115_20260905.json',
                  KD=ROOT / 'outputs/EVAL_SLOTDEEP_DISTILL_S1_STEP7115_20260906.json')


def canonical_target(path):
    path = Path(path)
    return (TAPS / path).resolve() if not path.is_absolute() else path.resolve()


def six_comparison(artifact, bootstrap=10000, seed=20260906):
    cal = json.loads(Path(artifact).read_text())
    assert cal['schema_version'] == 1 and cal['method'] == METHOD
    loaded, metadata, provenance_hashes = {}, {}, {}
    expected_keys, original_args = None, None
    for arm in RUNS:
        for calibrated in (False, True):
            label = arm + ('-calibrated' if calibrated else '-native')
            path = OUT / f'six_{arm}_calibrated.json' if calibrated else NATIVE_SIX[arm]
            data, arrays, keys = load(path, 'latgate')
            a = data['arguments']
            if expected_keys is None:
                expected_keys, original_args = keys, a
            assert expected_keys == keys, 'six-domain prompt pairing differs'
            for field in SEMANTICS:
                if field != 'gate_rho':
                    assert a[field] == original_args[field], f'six-domain semantics differ: {field}'
            assert a['max_samples'] == 20 and a['max_new_tokens'] == 256
            assert a['shuffle_seed'] == 2026 and a['eval_reserved'] is True
            assert canonical_target(a['target_model']) == canonical_target(cal['models'][arm]['target_model'])
            assert Path(a['lattice_head']).resolve() == (export_path(arm) / 'selector.pt').resolve()
            assert sha(a['lattice_head']) == cal['models'][arm]['selector_sha256']
            p = cal['models'][arm]['exported_decode_policy']
            assert a['gate_rho'] == (cal['models'][arm]['fit']['rho'] if calibrated else p['selector_gate_rho'])
            assert a['gate_tau'] == p['selector_gate_tau'] == 0
            assert a['gate_theta'] == p['selector_gate_theta'] == 0 and not a['gate_skip0']
            assert a['score_temperature'] == 1 and a['beam'] == a['path_branch'] == 1
            if calibrated:
                tag = f'six_{arm}_calibrated'
                exit_path = OUT / (tag + '_EXIT.json')
                sidecar = OUT / (tag + '_PROVENANCE.json')
                done = json.loads(exit_path.read_text())
                provenance = json.loads(sidecar.read_text())
                assert done['success'] is True
                c = provenance['calibration_provenance']
                assert c['artifact_sha256'] == sha(artifact) and c['arm'] == arm
                assert c['original_policy'] == p and c['fitted_rho'] == a['gate_rho']
                assert c['independent_prompt_check'] is True
                assert provenance['selector_sha256'] == cal['models'][arm]['selector_sha256']
                assert provenance['source_sha256'][str(TAPS / 'scripts/decode_lattice.py')] == cal['reference_decoder_sha256']
                for record in (sidecar, exit_path):
                    provenance_hashes[str(record)] = sha(record)
            loaded[label], metadata[label] = arrays, data
            provenance_hashes[str(path)] = sha(path)
    rng = np.random.default_rng(seed)
    draws = {d: rng.integers(0, 20, (bootstrap, 20)) for d in DOMAINS}
    results = []
    for label, arrays in loaded.items():
        m = metrics(arrays)
        results.append(dict(label=label, **m, reachable=diagnostics(metadata[label], 'latgate'),
            paired_comparisons={ref: dict(delta=m['macro']-metrics(base)['macro'],
                                          paired_95ci=bootstrap_delta(arrays, base, draws))
                                for ref, base in loaded.items()}))
    # Pre-existing known overlap; symmetric sensitivity, not a cleaned primary set.
    trimmed = {label: {d: a[np.arange(len(a)) != 7] if d == 'math500' else a
                        for d, a in groups.items()} for label, groups in loaded.items()}
    rng = np.random.default_rng(seed)
    draws = {d: rng.integers(0, 19 if d == 'math500' else 20,
                           (bootstrap, 19 if d == 'math500' else 20)) for d in DOMAINS}
    sensitivity = {label: dict(macro=metrics(groups)['macro'],
        paired_comparisons={ref: dict(delta=metrics(groups)['macro']-metrics(base)['macro'],
                                      paired_95ci=bootstrap_delta(groups, base, draws))
                            for ref, base in trimmed.items()}) for label, groups in trimmed.items()}
    return dict(metric='mean of six per-domain accepted_sum/num_blocks ratios', seed=seed,
                bootstrap=bootstrap, population=120, prompts_per_domain=20,
                scope='Previously used six-domain development set, not untouched confirmation; target-greedy agreement, not task accuracy or throughput',
                reachable_scope='Policy-dependent accepted prefix plus first rejection; aggregate ratios, no per-prompt ratio CI on this CLI output',
                native_provenance_limit='Native six-domain JSON is historical; current export identity plus its previously audited run are used, not newly rerun native six-domain outputs',
                prompt_keys=expected_keys, source_sha256=provenance_hashes, results=results,
                known_overlap_sensitivity=dict(excluded=['math500:7'], population=119,
                    scope='Only known overlap removed, not proof of comprehensive decontamination', results=sensitivity))


def paired_text(value):
    lo, hi = value['paired_95ci']
    return f"{value['delta']:+.5f} [{lo:+.5f}, {hi:+.5f}]"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-prefix', type=Path, required=True)
    args = parser.parse_args()
    jp, mp = args.out_prefix.with_suffix('.json'), args.out_prefix.with_suffix('.md')
    assert not jp.exists() and not mp.exists()
    specifications = [f"{a}-{'calibrated' if c else 'native'}={OUT}/confirmation_{a}_{'calibrated' if c else 'native'}.json"
                      for a in RUNS for c in (False, True)]
    pb = compare(specifications, 'CE-calibrated', control='CE-native', allow_calibrated_rho=True,
                 expected_prompts=512, bootstrap=10000, seed=20260906)
    six = six_comparison(ARTIFACT)
    payload = dict(calibration_artifact=str(ARTIFACT), calibration_sha256=sha(ARTIFACT),
                   analysis_source_sha256=sha(__file__), pb_confirmation=pb, six_development=six)
    contrasts = [('KD-calibrated', 'CE-calibrated'), ('CE-calibrated', 'CE-native'),
                 ('KD-calibrated', 'KD-native')]
    lines = ['# SlotDeep：冻结权重的独立门槛校准', '',
             '本轮只校准全局 rho；不新增参数、不训练、不改变目标模型或草稿候选。', '',
             '## 真实解码结果', '',
             '| 模型 | 新 PB confirmation | 六域 development macro |',
             '| --- | ---: | ---: |']
    by_pb = {r['label']:r for r in pb['results']}
    by_six = {r['label']:r for r in six['results']}
    for label, row in by_pb.items():
        lines.append(f"| {label} | {row['pooled_acceptance']:.5f} | {by_six[label]['macro']:.5f} |")
    lines += ['', '| 预声明对照 | PB 差值 [paired 95% CI] | 六域差值 [paired 95% CI] |',
              '| --- | --- | --- |']
    for left, right in contrasts:
        lines.append(f"| {left} − {right} | {paired_text(by_pb[left]['paired_comparisons'][right])} | {paired_text(by_six[left]['paired_comparisons'][right])} |")
    lines += ['', '## PB 原生可达修复与误改', '',
              '这里是各自策略的可达前缀+拒绝位置，不是同一固定 BASE 首错人口。', '',
              '| 模型 | 可修复位置修复率 % | 正确位置误改率 % |', '| --- | ---: | ---: |']
    for label, r in pb['reachable_diagnostics'].items():
        lines.append(f"| {label} | {r['repair_pct']:.4f} | {r['destroy_pct']:.4f} |")
    lines += ['', '## 解释边界', '',
              '- 配对单位是 prompt；10000 次按 PB 来源/六域分层 bootstrap，seed20260906，不含训练种子不确定性。',
              '- 校准经验误改约束不是部署风险保证；确认集结果未用于门槛拟合。',
              '- 六域每域20条为已使用开发集，另报对称剔除已知 math500:7 的119条敏感性结果（JSON）。',
              '- 接受长度是未裁剪 accept+1，不等于任务答案正确率、有效 token 吞吐或速度。',
              '- legacy 校准特征未记录采集时权重哈希，当前身份校验不能补造历史证据。', '']
    jp.parent.mkdir(parents=True, exist_ok=True)
    write_json(jp, payload)
    with mp.open('x') as sink:
        sink.write('\n'.join(lines))
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
