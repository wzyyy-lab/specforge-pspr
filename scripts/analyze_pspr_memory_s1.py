"""Paired, domain-stratified analysis of real decode_lattice JSON artifacts.

Acceptance is agreement with the greedy target model, NOT benchmark answer
accuracy. Resample prompts (not dependent slots/blocks) within each domain.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


DOMAINS = ('gsm8k', 'math500', 'humaneval', 'mbpp', 'alpaca', 'mt-bench')
SEMANTICS = ('max_samples', 'max_new_tokens', 'shuffle_seed', 'eval_reserved',
             'draft_attn', 'gate_rho', 'gate_tau', 'gate_theta', 'gate_skip0',
             'score_temperature', 'cloze_causal', 'cloze_anchor_self',
             'cloze_zero_context', 'cloze_zero_state', 'cloze_zero_direct_hidden')


def load(path, mode):
    data = json.loads(Path(path).read_text())
    assert data['block_size'] == 16 and data['proposal_slots'] == 15
    rows = [r for r in data['results'] if r['mode'] == mode]
    keys = [(r['dataset'], r['prompt_index'], r['prompt_sha256']) for r in rows]
    assert len(keys) == len(set(keys)) == 120, (path, len(keys))
    by_domain, prompt_keys = {}, {}
    for domain in DOMAINS:
        group = sorted((r for r in rows if r['dataset'] == domain),
                       key=lambda r: r['prompt_index'])
        assert len(group) == 20, (path, domain, len(group))
        for r in group:
            a = r['acceptance_lengths']
            assert a and all(isinstance(v, int) and 1 <= v <= 16 for v in a)
            assert sum(a) == r['accepted_sum'] and len(a) == r['num_blocks']
            assert abs(sum(a) / len(a) - r['mean_acceptance']) < 1e-10
        by_domain[domain] = np.array([[r['accepted_sum'], r['num_blocks']]
                                      for r in group], dtype=np.float64)
        prompt_keys[domain] = [r['prompt_sha256'] for r in group]
    return data, by_domain, prompt_keys


def metrics(arrays):
    domain = {d: float(a[:, 0].sum() / a[:, 1].sum()) for d, a in arrays.items()}
    merged = np.concatenate(list(arrays.values()))
    return dict(domains=domain, macro=float(np.mean(list(domain.values()))),
                micro=float(merged[:, 0].sum() / merged[:, 1].sum()))


def bootstrap_delta(arrays, baseline, draws):
    delta = np.zeros(next(iter(draws.values())).shape[0])
    for d in DOMAINS:
        x, b = arrays[d][draws[d]], baseline[d][draws[d]]
        delta += x[:, :, 0].sum(1) / x[:, :, 1].sum(1)
        delta -= b[:, :, 0].sum(1) / b[:, :, 1].sum(1)
    return np.quantile(delta / len(DOMAINS), [0.025, 0.975]).tolist()


def paired_control_comparison(loaded, reference, draws):
    """Compare predeclared recipes with a loaded, paired control, not just Cloze."""
    if reference is None:
        return {}
    assert reference in loaded, f'paired reference is not a result label: {reference}'
    base = loaded[reference]
    return {label: dict(reference=reference,
                        delta_macro=metrics(arrays)['macro'] - metrics(base)['macro'],
                        paired_95ci=bootstrap_delta(arrays, base, draws))
            for label, arrays in loaded.items() if label != reference}


def loss_pilot_description():
    return dict(
        experiment='SlotDeep training-only pilots; weights-only warm start from Stage1 step7115; fresh optimizer and scheduler',
        checkpoint_policy='All three recipes have a predeclared step1000 endpoint; no test-selected best checkpoint',
        scope_note='These are 1000-step weights-only warm-start pilots on the regenerated PerfectBlend corpus, '
                   'not whole-head Stage1 training from random initialization or a full online epoch. '
                   'The architecture and 47.153M selector parameter count are unchanged. '
                   'The untrimmed accept+1 metric is not a throughput or task-accuracy measurement.')


def turn_recipe_description():
    return dict(
        experiment='SlotDeep whole-selector online Stage1 from random initialization; turn-wise regenerated PerfectBlend recipe',
        checkpoint_policy='step1000 is interim; fixed-budget step7115 is the predeclared primary endpoint; '
                          'no test-selected best checkpoint. 7115 steps are NOT a full epoch of the 253480-turn pool.',
        scope_note='The 47.153M selector is randomly initialized and the official DFlash backbone is frozen. '
                   'The fixed 7115-step/global-batch28 budget consumes at most 199220 turn instances (78.59% of the pool). '
                   'The comparison changes turn sampling, prefix construction, content-only supervision, termination handling, '
                   'and precision/order settings; it is a data-recipe bundle, not a prefix-only ablation. '
                   'The untrimmed accept+1 metric is not a throughput or task-accuracy measurement.')


def diagnostics(data, mode):
    st = data.get('gate_stats', {}).get(mode)
    if st is None:
        return None
    assert st['fix_recovered'] + st['fix_kept_wrong'] + st['fix_wrong_override'] == st['fixable_n']
    assert st['base_right_kept'] + st['base_right_destroyed'] == st['base_right_n']
    assert st['fixable_n'] + st['unfixable_n'] == st['base_wrong_n']
    return dict(
        population='reachable accepted-prefix plus first-rejected position; policy-dependent',
        fixable_n=st['fixable_n'], recovered=st['fix_recovered'],
        reachable_repair_pct=100 * st['fix_recovered'] / st['fixable_n'],
        base_right_n=st['base_right_n'], destroyed=st['base_right_destroyed'],
        reachable_destroy_pct=100 * st['base_right_destroyed'] / st['base_right_n'],
        true_best_gate_blocked_pct=100 * st['fix_gate_blocked_true_best'] / st['fixable_n'],
    )


def exclusion_sensitivity(records, exclusions, bootstrap):
    """Drop explicitly named prompts symmetrically, without altering raw results.

    This is a known-overlap sensitivity analysis, NOT proof that remaining
    prompts are deduplicated. Gate counters are run-pooled, so they cannot be
    recomputed for this subset and are deliberately omitted.
    """
    excluded = set()
    for item in exclusions:
        domain, index = item.rsplit(':', 1)
        assert domain in DOMAINS, item
        excluded.add((domain, int(index)))
    arrays_by_model = []
    prompt_groups = None
    for record in records:
        data, _, _ = load(record['path'], record['mode'])
        available = {(r['dataset'], r['prompt_index']) for r in data['results']
                     if r['mode'] == record['mode']}
        assert excluded <= available, f'excluded prompt absent: {excluded - available}'
        grouped, keys = {}, {}
        for domain in DOMAINS:
            rows = sorted((r for r in data['results'] if r['mode'] == record['mode']
                           and r['dataset'] == domain
                           and (domain, r['prompt_index']) not in excluded),
                          key=lambda r: r['prompt_index'])
            assert rows, f'no prompts remain for {domain}'
            grouped[domain] = np.array([[r['accepted_sum'], r['num_blocks']] for r in rows],
                                       dtype=np.float64)
            keys[domain] = [(r['prompt_index'], r['prompt_sha256']) for r in rows]
        if prompt_groups is None:
            prompt_groups = keys
        assert prompt_groups == keys, 'sensitivity prompts differ between models'
        arrays_by_model.append(grouped)
    rng = np.random.default_rng(20260905)
    draws = {d: rng.integers(0, len(prompt_groups[d]), (bootstrap, len(prompt_groups[d])))
             for d in DOMAINS}
    baseline = arrays_by_model[1]
    baseline_macro = metrics(baseline)['macro']
    results = []
    for record, arrays in zip(records, arrays_by_model):
        m = metrics(arrays)
        results.append(dict(label=record['label'], **m,
                            delta_vs_cloze=m['macro'] - baseline_macro,
                            paired_95ci=bootstrap_delta(arrays, baseline, draws)))
    return dict(scope='known-overlap exclusion sensitivity; not a fully deduplicated held-out set',
                exclusions=sorted(exclusions),
                domain_prompt_counts={d: len(prompt_groups[d]) for d in DOMAINS},
                prompt_keys=prompt_groups, results=results,
                gate_diagnostics='unavailable for subset: raw gate counters are run-pooled')


def teacher_source_status(sources, require_match=False):
    valid = all(isinstance(h, str) and len(h) == 64 and
                all(c in '0123456789abcdef' for c in h) for h in sources.values())
    status = ('unavailable' if not sources or not valid else
              'matched' if len(set(sources.values())) == 1 else 'mixed')
    if require_match:
        assert status == 'matched', f'teacher diagnostic source hashes do not match: {sources}'
    return dict(status=status, diagnostic_source_sha256=sources,
                scope='entry-point source identity only; not complete feature-generation lineage')


def teacher_comparison(reference_path, result_specs, manifest_path, bootstrap,
                       require_source_match=False):
    """Paired prompt uncertainty for the separate, fixed HF teacher-prefix proxy."""
    manifest = json.loads(Path(manifest_path).read_text())
    assert len(manifest) == 120
    by_file = {row['file']: row for row in manifest}
    assert len(by_file) == 120
    specifications = [('Cloze7115 baseline', reference_path)]
    specifications += [item.split('=', 1) for item in result_specs]
    keys = ('fix_recovered', 'base_wrong_n', 'fixable_n', 'base_right_destroyed', 'base_right_n')
    fixed = ('decision_n', 'base_right_n', 'base_wrong_n', 'fixable_n', 'unfixable_n', 'covered_n')
    semantics = ('num_samples', 'num_anchors', 'chunk_blocks', 'max_length', 'seed', 'gate_rho',
                 'gate_tau', 'gate_theta', 'attention_backend', 'preserve_selector_fp32',
                 'rho_mode', 'rho_values')
    reference, reference_order = None, None
    models, records, hashes, script_sources = [], [], {}, {}
    for label, path in specifications:
        raw = Path(path).read_bytes()
        hashes[str(Path(path).resolve())] = hashlib.sha256(raw).hexdigest()
        data = json.loads(raw)
        script_sources[label] = data.get('input_provenance', {}).get('diagnostic_source_sha256')
        assert data['gates'] and all(data['gates'].values()), (label, 'diagnostic gates failed')
        assert data['arguments']['preserve_selector_fp32'] is True
        rows = data['first_base_error_per_sample']
        assert len(rows) == 120
        order = [(Path(r['feature_file']).name, r['feature_sha256']) for r in rows]
        assert len(set(order)) == 120 and {name for name, _ in order} == set(by_file)
        if reference is None:
            reference, reference_order = data, order
        assert order == reference_order, (label, 'feature order/hash mismatch')
        assert all(data['arguments'][k] == reference['arguments'][k] for k in semantics)
        grouped = {d: [] for d in DOMAINS}
        for row, ref in zip(rows, reference['first_base_error_per_sample']):
            assert all(row['base_counts'][k] == ref['base_counts'][k] for k in fixed)
            m = by_file[Path(row['feature_file']).name]
            grouped[m['dataset']].append((m['prompt_index'], [row['base_counts'][k] for k in keys]))
        arrays = {d: np.array([v for _, v in sorted(grouped[d])], dtype=np.float64) for d in DOMAINS}
        assert all(len(a) == 20 for a in arrays.values())
        totals = np.concatenate(list(arrays.values())).sum(0)
        counters = data['populations']['BASE']['counters']
        assert all(int(totals[i]) == counters[k] for i, k in enumerate(keys))
        records.append(dict(label=label, path=path, counters=counters,
                            diagnostic=data['first_base_error_diagnostic'],
                            ranking=data['populations']['BASE']['derived']))
        models.append(arrays)
    rng = np.random.default_rng(20260905)
    sampled = [np.zeros((bootstrap, len(keys))) for _ in models]
    for d in DOMAINS:
        draw = rng.integers(0, 20, (bootstrap, 20))
        for destination, arrays in zip(sampled, models):
            destination += arrays[d][draw].sum(1)
    for record, sample in zip(records, sampled):
        intervals = {}
        for name, numerator, denominator in [('repair_all_pp', 0, 1),
                                             ('repair_given_topk_pp', 0, 2),
                                             ('teacher_prefix_destroy_pp', 3, 4)]:
            difference = 100 * (sample[:, numerator] / sample[:, denominator]
                                - sampled[0][:, numerator] / sampled[0][:, denominator])
            intervals[name] = np.quantile(difference, [.025, .975]).tolist()
        record['paired_95ci_delta_vs_cloze_pp'] = intervals
    hashes[str(Path(manifest_path).resolve())] = hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest()
    return dict(population='separate HF-generated fixed teacher-prefix proxy; NOT decode trajectory replay',
                inference='pooled BASE count ratios; paired prompts resampled within each domain; no seed uncertainty',
                source_comparability=teacher_source_status(script_sources, require_source_match),
                source_sha256=hashes, results=records)


def risk_counts(kind, margin, slot, threshold):
    """Fixed BASE prefix diagnostic, not a replay after changing the policy.

    BASE contains the original correct prefix and at most its first error.
    Count a repaired terminal error as safe only if no earlier correct slot
    was changed. Entirely base-correct blocks have no repair opportunity.
    """
    assert kind.ndim == margin.ndim == slot.ndim == 1
    assert len(kind) == len(margin) == len(slot) and len(kind) > 0
    assert np.isfinite(margin).all() and np.isin(kind, [0, 1, 2, 3]).all()
    starts = np.flatnonzero(slot == 0)
    assert len(starts) and starts[0] == 0, 'BASE block does not begin at slot zero'
    ends = np.r_[starts[1:] - 1, len(slot) - 1]
    assert np.array_equal(slot, np.concatenate([np.arange(e - s + 1)
                                                for s, e in zip(starts, ends)]))
    terminal = np.zeros(len(kind), dtype=bool)
    terminal[ends] = True
    assert np.all(kind[~terminal] == 0), 'BASE contains a slot after its first error'
    replace = margin > threshold  # Strict comparison matches tau=theta=0 serving.
    harm = replace & (kind == 0)
    repair = replace & (kind == 1)
    damaged = np.add.reduceat(harm.astype(np.int64), starts) > 0
    return dict(repaired=int(repair.sum()), destroyed=int(harm.sum()),
                safe_first_repairs=int((repair[ends] & ~damaged).sum()),
                damaged_prefix_blocks=int(damaged.sum()),
                damaged_before_first_error_blocks=int((damaged & (kind[ends] != 0)).sum()),
                damaged_fully_base_correct_blocks=int((damaged & (kind[ends] == 0)).sum()),
                blocks=len(starts))


def risk_budget_threshold(margin, kind, budget):
    """Largest strict-threshold repair set with at most budget observed harms.

    Tied negatives may leave budget unused. This selects with the very labels
    being reported: an optimistic diagnostic envelope, NOT held-out calibration.
    """
    negative = np.sort(margin[kind == 0].astype(np.float64))[::-1]
    assert 0 <= budget < len(negative), 'risk budget outside observed negatives'
    return float(negative[budget])


def risk_comparison(specifications, budgets, bootstrap):
    models, records, hashes = [], [], {}
    semantics = ('num_samples', 'num_anchors', 'chunk_blocks', 'max_length', 'seed',
                 'gate_rho', 'gate_tau', 'gate_theta', 'attention_backend',
                 'preserve_selector_fp32', 'rho_mode', 'rho_values', 'features')
    reference = None
    for item in specifications:
        label, raw_path = item.split('=', 1)
        path = Path(raw_path).resolve()
        meta = json.loads(path.with_suffix('.json').read_text())
        args = meta['arguments']
        assert meta['gates'] and all(meta['gates'].values()), label
        assert args['preserve_selector_fp32'] and args['include_base_decisions']
        assert args['gate_tau'] == args['gate_theta'] == 0 and args['gate_rho'] > 0
        assert args['rho_mode'] == 'global'
        if reference is None:
            reference = args
        assert all(args[k] == reference[k] for k in semantics), 'risk semantics differ'
        with np.load(path, allow_pickle=False) as data:
            arrays = {k: data['BASE_' + k].copy() for k in ('kind', 'margin', 'slot', 'sample')}
            arrays['files'] = data['files'].copy()
        kind, margin, slot = (arrays[k] for k in ('kind', 'margin', 'slot'))
        assert len(arrays['sample']) == len(kind)
        assert len(arrays['files']) == 120 and len(set(arrays['files'])) == 120
        assert np.array_equal(np.unique(arrays['sample']), np.arange(120))
        if models:
            for k in ('slot', 'sample', 'files'):
                assert np.array_equal(arrays[k], models[0][k]), ('BASE ordering differs', k)
            for value in (0, 3):
                assert np.array_equal(kind == value, models[0]['kind'] == value)
        c = meta['populations']['BASE']['counters']
        assert int((kind == 0).sum()) == c['base_right_n']
        assert int(((kind == 1) | (kind == 2)).sum()) == c['fixable_n']
        assert int((kind != 0).sum()) == c['base_wrong_n']
        assert len(kind) == c['decision_n']
        assert c['fixable_n'] > 0 and c['base_right_n'] > 0
        rows = []
        thresholds = [('original_rho', np.log(args['gate_rho']), None)]
        thresholds += [('same_data_budget', risk_budget_threshold(margin, kind, b), b)
                       for b in budgets]
        for selection, threshold, budget in thresholds:
            counts = risk_counts(kind, margin, slot, threshold)
            if budget is None:
                assert counts['repaired'] == c['fix_recovered']
                assert counts['destroyed'] == c['base_right_destroyed']
            else:
                assert counts['destroyed'] <= budget
            rows.append(dict(selection=selection, budget=budget, log_margin_threshold=float(threshold),
                             **counts, repair_given_topk_pct=100 * counts['repaired'] / c['fixable_n'],
                             prefix_destroy_pct=100 * counts['destroyed'] / c['base_right_n']))
        records.append(dict(label=label, path=str(path), base_counters=c, rows=rows,
                            best_alternative_correct=int((kind == 1).sum()),
                            best_alternative_correct_pct=100 * int((kind == 1).sum()) / c['fixable_n']))
        for source in (path, path.with_suffix('.json')):
            hashes[str(source)] = hashlib.sha256(source.read_bytes()).hexdigest()
        models.append(arrays)
    assert len(models) >= 2, 'risk comparison needs a reference and at least one comparator'
    assert len({r['label'] for r in records}) == len(records)
    # Alternative-ranking uncertainty uses whole prompts, never dependent anchors.
    domains = [Path(str(f)).stem.rsplit('_', 1)[0] for f in models[0]['files']]
    assert set(domains) == set(DOMAINS)
    rng = np.random.default_rng(20260905)
    totals = [np.zeros((bootstrap, 2)) for _ in models]
    for domain in DOMAINS:
        indices = [i for i, d in enumerate(domains) if d == domain]
        assert len(indices) == 20
        draw = rng.integers(0, 20, (bootstrap, 20))
        for destination, model in zip(totals, models):
            per_prompt = []
            for index in indices:
                k = model['kind'][model['sample'] == index]
                per_prompt.append([int((k == 1).sum()), int(((k == 1) | (k == 2)).sum())])
            destination += np.asarray(per_prompt)[draw].sum(1)
    for record, total in zip(records, totals):
        delta = 100 * (total[:, 0] / total[:, 1] - totals[0][:, 0] / totals[0][:, 1])
        record['best_alternative_delta_95ci_pp'] = np.quantile(delta, [.025, .975]).tolist()
        for row, ref in zip(record['rows'], records[0]['rows']):
            row['repair_delta_vs_reference_pp'] = row['repair_given_topk_pct'] - ref['repair_given_topk_pct']
    return dict(population='fixed BASE on separate HF teacher prefixes, not decode trajectory replay',
                threshold_selection='same-data optimistic risk envelope; NOT independent calibration or a serving recommendation',
                inference='no risk-curve confidence guarantee; alternative-ranking CI resamples paired prompts within domains',
                provenance='hashes pin existing NPZ/JSON; this is post-hoc analysis, not a prelaunch feature witness',
                source_sha256=hashes, results=records)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--baseline', required=True)
    p.add_argument('--result', action='append', required=True, help='LABEL=PATH')
    p.add_argument('--out-prefix', required=True)
    p.add_argument('--bootstrap', type=int, default=10000)
    p.add_argument('--run-kind', choices=('memory', 'slotdeep', 'slotdeep-loss', 'slotdeep-turn'), default='memory',
                   help='report scope only; preserves historical Memory analysis by default')
    p.add_argument('--paired-reference', help='exact --result LABEL for paired recipe comparisons, e.g. T0')
    p.add_argument('--sensitivity-exclude-prompt', action='append', default=[],
                   help='DATASET:INDEX; additional symmetric subset analysis, never changes primary 120 prompts')
    p.add_argument('--teacher-reference')
    p.add_argument('--teacher-result', action='append', default=[], help='LABEL=PATH')
    p.add_argument('--require-matched-teacher-source', action='store_true',
                   help='reject fixed-teacher comparisons with missing or unequal diagnostic source hashes')
    p.add_argument('--feature-manifest')
    p.add_argument('--risk-decisions', action='append', default=[], help='LABEL=NPZ; reference first')
    p.add_argument('--risk-budgets', default='0,156,312,328,469,781,966',
                   help='observed damage-count budgets; exploratory same-data envelope only')
    args = p.parse_args()
    if args.require_matched_teacher_source and not args.teacher_reference:
        p.error('--require-matched-teacher-source requires --teacher-reference')
    bdata, base, keys = load(args.baseline, 'latgate')
    rng = np.random.default_rng(20260905)
    draws = {d: rng.integers(0, 20, size=(args.bootstrap, 20)) for d in DOMAINS}
    records = []
    for label, mode in [('DFlash backbone', 'oneshot'), ('Cloze7115 baseline', 'latgate'),
                        ('Oracle16 privileged', 'oracle16')]:
        data, arrays, ref_keys = load(args.baseline, mode)
        assert keys == ref_keys
        records.append(dict(label=label, path=args.baseline, mode=mode,
                            **metrics(arrays), diagnostics=diagnostics(data, mode)))
    loaded = {}
    for item in args.result:
        label, path = item.split('=', 1)
        assert label not in loaded
        data, arrays, actual_keys = load(path, 'latgate')
        assert keys == actual_keys, f'prompt mismatch: {label}'
        for k in SEMANTICS:
            assert data['arguments'][k] == bdata['arguments'][k], (label, k)
        m = metrics(arrays)
        records.append(dict(label=label, path=path, mode='latgate', **m,
                            delta_vs_cloze=m['macro'] - records[1]['macro'],
                            paired_95ci=bootstrap_delta(arrays, base, draws),
                            diagnostics=diagnostics(data, 'latgate')))
        loaded[label] = arrays
    matched_controls = {}
    for label, arrays in loaded.items():
        if label.startswith('verified-'):
            control = label.replace('verified-', 'blind-', 1)
            if control in loaded:
                matched_controls[label] = dict(
                    delta_macro=metrics(arrays)['macro'] - metrics(loaded[control])['macro'],
                    paired_95ci=bootstrap_delta(arrays, loaded[control], draws),
                )
    paired_controls = paired_control_comparison(loaded, args.paired_reference, draws)
    hashes = {str(p): hashlib.sha256(Path(p).read_bytes()).hexdigest()
              for p in {r['path'] for r in records}}
    payload = dict(metric='macro of six domain accepted_sum / num_blocks',
                   scope='single training seed, six fixed domains x 20 prompts, greedy target agreement',
                   labels='target-model reference, not dataset task-answer correctness',
                   inference='95% paired within-domain prompt bootstrap; not training-seed uncertainty; no multiplicity adjustment',
                   diagnostics='reachable repair rates are not strictly first-base-error rates; population changes with policy',
                   source_sha256=hashes, results=records, matched_controls=matched_controls,
                   paired_recipe_controls=paired_controls)
    if args.run_kind == 'slotdeep':
        payload['experiment'] = 'SlotDeep in-place optimization; whole-head online Stage1 from scratch'
        payload['checkpoint_policy'] = ('step1000 is interim; full-epoch step7115 is the predeclared primary endpoint; '
                                        'no test-selected best checkpoint')
        payload['metric_boundary'] = ('accepted draft prefix + one target token; final blocks are not trimmed '
                                     'at EOS/max-new-tokens, so this is not effective-token throughput')
        payload['analysis_source_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if args.run_kind == 'slotdeep-loss':
        payload.update(loss_pilot_description())
        payload['metric_boundary'] = ('accepted draft prefix + one target token; final blocks are not trimmed '
                                     'at EOS/max-new-tokens, so this is not effective-token throughput')
        payload['analysis_source_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if args.run_kind == 'slotdeep-turn':
        payload.update(turn_recipe_description())
        payload['metric_boundary'] = ('accepted draft prefix + one target token; final blocks are not trimmed '
                                     'at EOS/max-new-tokens, so this is not effective-token throughput')
        payload['analysis_source_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if args.sensitivity_exclude_prompt:
        payload['known_overlap_sensitivity'] = exclusion_sensitivity(
            records, args.sensitivity_exclude_prompt, args.bootstrap)
    if args.teacher_reference or args.teacher_result or args.feature_manifest:
        assert args.teacher_reference and args.teacher_result and args.feature_manifest
        payload['fixed_teacher_prefix_diagnostic'] = teacher_comparison(
            args.teacher_reference, args.teacher_result, args.feature_manifest, args.bootstrap,
            require_source_match=args.require_matched_teacher_source)
    if args.risk_decisions:
        payload['matched_risk_diagnostic'] = risk_comparison(
            args.risk_decisions, [int(x) for x in args.risk_budgets.split(',')], args.bootstrap)
    prefix = Path(args.out_prefix)
    jp, mp = prefix.with_suffix('.json'), prefix.with_suffix('.md')
    assert not jp.exists() and not mp.exists(), 'refusing to overwrite analysis'
    jp.parent.mkdir(parents=True, exist_ok=True)
    jp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n')
    title = ('PSPR Memory Stage 1' if args.run_kind == 'memory' else 'PSPR SlotDeep Stage 1')
    if args.run_kind == 'slotdeep-loss':
        title = 'PSPR SlotDeep training-only warm-start pilots'
    if args.run_kind == 'slotdeep-turn':
        title = 'PSPR SlotDeep turn-wise data recipe Stage 1'
    lines = [f'# {title}: six-domain paired evaluation', '',
             'Target-greedy acceptance, not GSM8K/math/code task correctness. Six domains × 20 fixed prompts; H=15, K=16, max_new_tokens=256, shuffle seed 2026.', '',
             '| Model | Macro acceptance | Δ vs Cloze | Paired 95% CI | Reachable repair % | Reachable destroy % |',
             '| --- | ---: | ---: | --- | ---: | ---: |']
    for r in records:
        d = r.get('diagnostics') or {}
        delta = f'{r["delta_vs_cloze"]:+.5f}' if 'delta_vs_cloze' in r else '—'
        ci = '[%.5f, %.5f]' % tuple(r['paired_95ci']) if 'paired_95ci' in r else '—'
        repair = f'{d["reachable_repair_pct"]:.3f}' if d else '—'
        destroy = f'{d["reachable_destroy_pct"]:.3f}' if d else '—'
        lines.append(f'| {r["label"]} | {r["macro"]:.5f} | {delta} | {ci} | {repair} | {destroy} |')
    checkpoint_note = ('All checkpoints are reported. The best of four checkpoints is exploratory selection on these 120 prompts, not independent test confirmation.'
                       if args.run_kind == 'memory' else
                       'Step1000 is an interim diagnostic; the full-epoch step7115 is the predeclared primary endpoint, not a test-selected best checkpoint.')
    if args.run_kind in ('slotdeep-loss', 'slotdeep-turn'):
        checkpoint_note = payload['checkpoint_policy']
    lines += ['', 'Reachable diagnostics cover accepted-prefix positions plus the first rejected position. These are NOT strictly one first-base-error trial per block. Denominators are policy-dependent.', '',
              checkpoint_note + ' Confidence intervals do not include training-seed variability or multiple-comparison adjustment.', '',
              '## Per-domain acceptance', '',
              '| Model | ' + ' | '.join(DOMAINS) + ' |',
              '| --- | ' + ' | '.join(['---:'] * len(DOMAINS)) + ' |']
    for r in records:
        lines.append('| ' + r['label'] + ' | ' + ' | '.join(f'{r["domains"][d]:.5f}' for d in DOMAINS) + ' |')
    if paired_controls:
        lines += ['', '## Paired recipe comparison', '',
                  f'Reference: {args.paired_reference}. Same prompts and endpoint; confidence intervals do not include training-seed variability.', '',
                  '| Model | Δ macro vs reference | Paired 95% CI |', '| --- | ---: | --- |']
        for label, r in paired_controls.items():
            lines.append(f'| {label} | {r["delta_macro"]:+.5f} | '
                         f'[{r["paired_95ci"][0]:.5f}, {r["paired_95ci"][1]:.5f}] |')
    sensitivity = payload.get('known_overlap_sensitivity')
    if sensitivity:
        lines += ['', '## Known-overlap exclusion sensitivity', '',
                  'Excluded symmetrically: ' + ', '.join(sensitivity['exclusions']) + '. '
                  'This does not prove all other prompts are contamination-free. Primary results above remain unchanged.', '',
                  '| Model | Macro acceptance | Delta vs Cloze | Paired 95% CI |',
                  '| --- | ---: | ---: | --- |']
        for r in sensitivity['results']:
            lines.append(f'| {r["label"]} | {r["macro"]:.5f} | {r["delta_vs_cloze"]:+.5f} | '
                         f'[{r["paired_95ci"][0]:.5f}, {r["paired_95ci"][1]:.5f}] |')
    teacher = payload.get('fixed_teacher_prefix_diagnostic')
    if teacher:
        source_status = teacher['source_comparability']['status']
        lines += ['', '## Fixed first-base-error diagnostic', '',
                  f'Diagnostic entry-point source status: {source_status}. '
                  'A mixed or unavailable source is not a controlled code-version comparison; '
                  'matched entry-point source alone does not establish complete feature-generation provenance.', '',
                  'Separate HF-generated teacher-prefix proxy, not a replay of the decode trajectories. '
                  'All 120 feature hashes and per-prompt BASE denominators match. Rates below pool counts; '
                  'uncertainty resamples prompts within domains, not dependent anchors.', '',
                  '| Model | First-error repair % | Repair given top16 % | Prefix destroy % |',
                  '| --- | ---: | ---: | ---: |']
        for r in teacher['results']:
            d = r['diagnostic']
            lines.append(f'| {r["label"]} | {100*d["repair_all"]:.3f} | '
                         f'{100*d["repair_given_topk"]:.3f} | {100*d["teacher_prefix_destroy_rate"]:.3f} |')
        lines += ['', 'Paired 95% intervals for rate differences versus Cloze, in percentage points:', '',
                  '| Model | Conditional repair difference | Prefix destroy difference |',
                  '| --- | --- | --- |']
        for r in teacher['results'][1:]:
            ci = r['paired_95ci_delta_vs_cloze_pp']
            repair, destroy = ci['repair_given_topk_pp'], ci['teacher_prefix_destroy_pp']
            lines.append(f'| {r["label"]} | [{repair[0]:+.3f}, {repair[1]:+.3f}] | '
                         f'[{destroy[0]:+.3f}, {destroy[1]:+.3f}] |')
    risk = payload.get('matched_risk_diagnostic')
    if risk:
        lines += ['', '## Fixed-BASE matched-risk diagnostic', '',
                  'Reanalysis of saved scores, not fresh decoding. Thresholds use these same labels: '
                  'an optimistic diagnostic envelope, NOT held-out calibration. No serving threshold was changed. '
                  'Zero observed harm is not a zero-risk guarantee.', '',
                  '| Model | Threshold selection / damage budget | Repairs / covered first errors | Damage / base-correct prefix slots | Safe first repairs |',
                  '| --- | --- | ---: | ---: | ---: |']
        for r in risk['results']:
            c = r['base_counters']
            for row in r['rows']:
                selection = 'original rho' if row['budget'] is None else str(row['budget'])
                lines.append(f'| {r["label"]} | {selection} | {row["repaired"]}/{c["fixable_n"]} '
                             f'({row["repair_given_topk_pct"]:.4f}%) | {row["destroyed"]}/{c["base_right_n"]} '
                             f'({row["prefix_destroy_pct"]:.4f}%) | {row["safe_first_repairs"]} |')
        lines += ['', 'Safe first repairs additionally require no damage anywhere before that first base error; '
                  'they still use fixed HF blocks and are not deployed acceptance.', '',
                  '| Model | Correct best alternative / covered first errors | Difference vs reference 95% CI (pp) |',
                  '| --- | ---: | --- |']
        for r in risk['results']:
            ci = r['best_alternative_delta_95ci_pp']
            lines.append(f'| {r["label"]} | {r["best_alternative_correct"]}/{r["base_counters"]["fixable_n"]} '
                         f'({r["best_alternative_correct_pct"]:.4f}%) | [{ci[0]:+.4f}, {ci[1]:+.4f}] |')
    if args.run_kind == 'memory':
        lines += ['', '## Verified memory versus same-step blind control', '',
                  '| Step | Δ macro | Paired 95% CI |', '| --- | ---: | --- |']
    for k, r in matched_controls.items():
        lines.append(f'| {k} | {r["delta_macro"]:+.5f} | [{r["paired_95ci"][0]:.5f}, {r["paired_95ci"][1]:.5f}] |')
    scope_note = ('This is a 1000-step residual-only pilot, not complete training on the ~200k regenerated corpus.'
                  if args.run_kind == 'memory' else
                  'SlotDeep trains the entire 47.153M selector from scratch with a frozen official DFlash backbone. Only a completed step7115 checkpoint represents the full online epoch; an interim checkpoint does not. The untrimmed accept+1 metric is not a throughput or task-accuracy measurement.')
    if args.run_kind in ('slotdeep-loss', 'slotdeep-turn'):
        scope_note = payload['scope_note']
    lines += ['', 'Frozen-backbone checks are recorded in the individual EXPORT logs; all 58 tensors match the official DFlash template. Original cloze/PSPR source files were not edited for this experiment.', '',
              scope_note + ' Stage 2 requires positive independent acceptance evidence.', '']
    mp.write_text('\n'.join(lines))
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
