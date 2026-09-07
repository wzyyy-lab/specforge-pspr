"""Paired prompt bootstrap for the predeclared independent PB validation split.

Primary PB acceptance pools accepted_sum / num_blocks across its fixed mixture.
This is not the six-domain macro, task accuracy, or effective-token throughput.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def calibration_origin(data):
    """Verify the explicit rho-only exception against its immutable artifact."""
    policy = data['decode_policy']
    c = data.get('calibration_provenance')
    if c is None:
        return policy
    assert c['method'] == 'pspr_global_rho_v1' and c['independent_prompt_check'] is True
    raw = Path(c['artifact']).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == c['artifact_sha256'], 'calibration artifact drift'
    artifact = json.loads(raw)
    assert artifact['method'] == c['method'] and artifact['schema_version'] == 1
    assert artifact['score_temperature'] == 1.0
    record = artifact['models'][c['arm']]
    assert record['selector_sha256'] == data['selector_sha256'], 'calibration selector mismatch'
    assert artifact['reference_decoder_sha256'] == data['reference_decoder_sha256']
    assert artifact['calibration']['prompts_sha256'] == c['calibration_prompts_sha256']
    original = data['exported_decode_policy']
    assert original == record['exported_decode_policy'] == c['original_policy']
    expected = {**original, 'selector_gate_rho': record['fit']['rho']}
    assert policy == expected, 'calibration may change only rho'
    assert policy['selector_gate_rho'] == c['fitted_rho']
    assert math.isfinite(c['fitted_rho']) and c['fitted_rho'] > 0
    return original


def compare(specs, reference, *, control=None, bootstrap=10000, expected_prompts=512,
            allow_calibrated_rho=False, seed=20260905):
    assert bootstrap > 0 and expected_prompts > 0
    labels, models, hashes = [], {}, {}
    first, first_keys, first_skips = None, None, None
    for spec in specs:
        label, path = spec.split('=', 1)
        assert label not in models, f'duplicate model label: {label}'
        raw = Path(path).read_bytes()
        data = json.loads(raw)
        assert data['block_size'] == 16 and data['proposal_slots'] == 15
        rows = sorted(data['results'], key=lambda r: r['prompt_index'])
        skipped = sorted(data['skipped'], key=lambda r: r['index'])
        assert len(rows) + len(skipped) == expected_prompts
        assert {r['prompt_index'] for r in rows} | {r['index'] for r in skipped} == set(range(expected_prompts))
        assert all(r['reason'] == 'prompt_token_limit' for r in skipped)
        keys = [(r['prompt_index'], r['id'], r['source'], r['prompt_sha256']) for r in rows]
        assert len(keys) == len(set(keys)) and len({r['id'] for r in rows}) == len(rows)
        for r in rows:
            a = r['acceptance_lengths']
            assert a and all(isinstance(x, int) and 1 <= x <= 16 for x in a)
            assert sum(a) == r['accepted_sum'] and len(a) == r['num_blocks']
            assert abs(r['mean_acceptance'] - sum(a)/len(a)) < 1e-10
        if first is None:
            first, first_keys, first_skips = data, keys, skipped
        assert keys == first_keys, 'holdout prompt identity/order differs'
        assert skipped == first_skips, 'holdout skipped population differs'
        for field in ('max_new_tokens', 'max_prompt_tokens', 'target_model'):
            assert data['arguments'][field] == first['arguments'][field], f'argument differs: {field}'
        for field in ('prompts_sha256', 'reference_decoder_sha256'):
            assert data[field] == first[field], f'evaluation semantics differ: {field}'
        if allow_calibrated_rho:
            assert calibration_origin(data) == calibration_origin(first), 'exported native policy differs'
        else:
            assert data['decode_policy'] == first['decode_policy'], 'evaluation semantics differ: decode_policy'
        models[label] = rows
        labels.append(label)
        hashes[str(Path(path).resolve())] = hashlib.sha256(raw).hexdigest()
    assert reference in models, 'reference label is absent'
    assert control is None or control in models, 'control label is absent'
    sources = sorted({r['source'] for r in models[reference]})
    rng = np.random.default_rng(seed)
    draws, arrays = {}, {}
    for source in sources:
        n = sum(r['source'] == source for r in models[reference])
        draws[source] = rng.integers(0, n, size=(bootstrap, n))
    for label, rows in models.items():
        arrays[label] = {s: np.array([[r['accepted_sum'], r['num_blocks']]
                                     for r in rows if r['source'] == s], dtype=np.float64)
                         for s in sources}
    rates, boot_rates = {}, {}
    for label, groups in arrays.items():
        summed = sum((a.sum(0) for a in groups.values()))
        rates[label] = float(summed[0] / summed[1])
        boot = sum((a[draws[s]].sum(1) for s, a in groups.items()))
        boot_rates[label] = boot[:, 0] / boot[:, 1]
    results = []
    for label in labels:
        row = dict(label=label, pooled_acceptance=rates[label],
                   per_source={s: float(a[:, 0].sum()/a[:, 1].sum()) for s, a in arrays[label].items()},
                   paired_comparisons={})
        for ref in (labels if allow_calibrated_rho else dict.fromkeys([reference] + ([control] if control else []))):
            row['paired_comparisons'][ref] = dict(delta=rates[label]-rates[ref],
                paired_95ci=np.quantile(boot_rates[label]-boot_rates[ref], [.025, .975]).tolist())
        results.append(row)
    reachable = {}
    if all(all('gate_stats' in r for r in rows) for rows in models.values()):
        keys = ('fix_recovered', 'fixable_n', 'base_right_destroyed', 'base_right_n')
        sampled, observed = {}, {}
        for label, rows in models.items():
            groups = {s: np.array([[r['gate_stats'].get(k, 0) for k in keys]
                                  for r in rows if r['source'] == s], dtype=np.float64) for s in sources}
            observed[label] = sum(a.sum(0) for a in groups.values())
            sampled[label] = sum(a[draws[s]].sum(1) for s, a in groups.items())
        for label in labels:
            total = observed[label]
            if total[1] <= 0 or total[3] <= 0:
                raise ValueError('reachable diagnostics require nonempty denominators')
            detail = dict(counters=dict(zip(keys, map(int, total))),
                          repair_pct=100*total[0]/total[1], destroy_pct=100*total[2]/total[3],
                          paired_comparisons={})
            for ref in (labels if allow_calibrated_rho else dict.fromkeys([reference] + ([control] if control else []))):
                detail['paired_comparisons'][ref] = {}
                for name, n, d in [('repair_pp', 0, 1), ('destroy_pp', 2, 3)]:
                    assert (sampled[label][:, d] > 0).all() and (sampled[ref][:, d] > 0).all()
                    delta = 100*(sampled[label][:, n]/sampled[label][:, d]
                                 - sampled[ref][:, n]/sampled[ref][:, d])
                    detail['paired_comparisons'][ref][name] = dict(
                        delta=100*(total[n]/total[d]-observed[ref][n]/observed[ref][d]),
                        paired_95ci=np.quantile(delta, [.025,.975]).tolist())
            reachable[label] = detail
    output = dict(metric='pooled accepted_sum / num_blocks over the fixed PB source mixture',
                scope='predeclared prompt split, exact normalized exclusion from declared head-training files; not semantic or pretraining decontamination',
                inference='paired prompt bootstrap stratified by PB source; no training-seed uncertainty or multiplicity adjustment',
                labels='target-greedy agreement; not task-answer accuracy or effective-token throughput',
                population=len(first_keys), skipped=first_skips,
                source_counts={s: len(arrays[reference][s]) for s in sources},
                prompts_sha256=first['prompts_sha256'], source_sha256=hashes,
                reference=reference, control=control, bootstrap=bootstrap, results=results)
    if allow_calibrated_rho:
        output.update(seed=seed, allow_calibrated_rho=True, reachable_diagnostics=reachable,
                      reachable_scope='policy-dependent accepted prefix plus first rejection; NOT fixed BASE first errors')
    return output


def report_description(run_kind):
    if run_kind == 'slotdeep-calibration':
        return ('SlotDeep: independent global-rho calibration and real decoding',
                'Both heads remain frozen. Calibrated rules change only global rho and are '
                'pinned before evaluation; no threshold is selected using these decode labels. '
                'Calibration empirical damage is not a deployment-risk guarantee. '
                'Results measure target-greedy agreement, not task accuracy or speed.')
    if run_kind == 'slotdeep-turn':
        return (
            'SlotDeep turn-wise Stage1: previously consumed PB validation cohort',
            'OldData7115 and TurnData7115 are random-initialization whole-selector '
            'Stage1 runs with a frozen official DFlash backbone, not weights-only '
            'warm-start pilots. The turn-wise run uses a fixed 7115-step budget, '
            'not a full epoch of the 253480-turn pool. This 512-prompt cohort has '
            'already been inspected for prior recipes and is development evidence, '
            'not untouched confirmation. No threshold was fitted to these labels.',
        )
    if run_kind == 'loss-pilot':
        return (
            'SlotDeep training-only pilots: independent PB validation',
            'T0/T1/T2 are fixed step1000 weights-only warm-start pilots with fresh '
            'optimizers, not random-initialization full Stage1 runs. No threshold '
            'was fitted to these validation labels.',
        )
    raise ValueError(f'unknown run kind: {run_kind}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result', action='append', required=True, help='LABEL=JSON')
    parser.add_argument('--reference', default='Cloze7115')
    parser.add_argument('--control', default='T0')
    parser.add_argument('--out-prefix', type=Path, required=True)
    parser.add_argument('--bootstrap', type=int, default=10000)
    parser.add_argument('--run-kind', choices=('loss-pilot', 'slotdeep-turn', 'slotdeep-calibration'),
                        default='loss-pilot', help='report wording only; no metric change')
    args = parser.parse_args()
    jp, mp = args.out_prefix.with_suffix('.json'), args.out_prefix.with_suffix('.md')
    assert not jp.exists() and not mp.exists(), 'refusing to overwrite analysis'
    result = compare(args.result, args.reference, control=args.control, bootstrap=args.bootstrap,
                     allow_calibrated_rho=args.run_kind == 'slotdeep-calibration',
                     seed=20260906 if args.run_kind == 'slotdeep-calibration' else 20260905)
    result['analysis_source_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    title, scope_note = report_description(args.run_kind)
    result['run_kind'] = args.run_kind
    result['scope_note'] = scope_note
    refs = list(dict.fromkeys([args.reference, args.control]))
    lines = ['# ' + title, '',
             result['metric'] + '.', '', result['labels'] + '.', '',
             f"Evaluated {result['population']} prompts; skipped {len(result['skipped'])} by the predeclared token limit.", '',
             '| Model | Acceptance | ' + ' | '.join(f'Δ vs {r} [paired 95% CI]' for r in refs) + ' |',
             '| --- | ---: | ' + ' | '.join('---' for _ in refs) + ' |']
    for row in result['results']:
        cells = [row['label'], f"{row['pooled_acceptance']:.5f}"]
        for ref in refs:
            p = row['paired_comparisons'][ref]
            cells.append(f"{p['delta']:+.5f} [{p['paired_95ci'][0]:+.5f}, {p['paired_95ci'][1]:+.5f}]")
        lines.append('| ' + ' | '.join(cells) + ' |')
    lines += ['', result['scope'] + '.', '', result['inference'] + '.', '',
              scope_note, '']
    jp.parent.mkdir(parents=True, exist_ok=True)
    with jp.open('x') as sink:
        json.dump(result, sink, indent=2)
        sink.write('\n')
    with mp.open('x') as sink:
        sink.write('\n'.join(lines))
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
