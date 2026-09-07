"""Verify a frozen-Stage2 continuation checkpoint, without modifying weights."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import torch

from scripts.run_slotdeep_reachable_continuation import INIT, build_config


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arm', choices=['control', 'reachable'], required=True)
    p.add_argument('--phase', choices=['smoke', 'train'], required=True)
    args = p.parse_args()
    cfg, _, run = build_config(args.arm, args.phase)
    step = cfg['training']['max_steps']
    path = Path(cfg['output_dir']) / f'{cfg["run_id"]}-step{step}' / 'training_state.pt'
    out = run / 'checkpoint_verification.json'
    if out.exists():
        raise FileExistsError(out)
    assert json.loads((run/'completion.json').read_text())['returncode'] == 0
    torch.set_num_threads(4)
    before = torch.load(INIT, map_location='cpu', weights_only=False, mmap=True)
    after = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    assert before['global_step'] == 9052 and after['global_step'] == step
    assert after['effective_total_steps'] == step
    assert after['world_size'] == 3 and after['batch_size'] == 2 and after['accumulation_steps'] == 4
    assert after['dflash2_selector_weight_mode'] == cfg['training']['dflash2_selector_weight_mode']
    assert after['dflash2_selector_survival_floor'] == .25
    assert after['dflash2_selector_gate_rho'] == 3
    assert after['dflash2_selector_gate_tau'] == after['dflash2_selector_gate_theta'] == 0
    assert after['pspr_slotdeep_reference_config'] == before['pspr_slotdeep_reference_config']
    initial, current = before['draft_state_dict'], after['draft_state_dict']
    assert initial.keys() == current.keys()
    groups = {}
    for name, value in current.items():
        assert value.shape == initial[name].shape
        assert torch.isfinite(value).all(), name
        kind = 'selector' if name.startswith('candidate_selector.') else 'backbone'
        group = groups.setdefault(kind, dict(tensors=0, elements=0, changed_tensors=0, dtypes=set()))
        group['tensors'] += 1
        group['elements'] += value.numel()
        group['changed_tensors'] += not torch.equal(value, initial[name])
        group['dtypes'].add(str(value.dtype))
    assert groups['backbone']['elements'] == 537427200 and groups['backbone']['changed_tensors'] == 0
    assert groups['selector']['elements'] == 47152913 and groups['selector']['changed_tensors'] > 0
    assert groups['selector']['dtypes'] == {'torch.float32'}
    for group in groups.values():
        group['dtypes'] = sorted(group['dtypes'])
    opt = after['replicated_optimizer_state']
    masters = opt['fp32_params']
    assert len(masters) == 104 and sum(x.numel() for x in masters) == 47152913
    assert all(x.dtype == torch.float32 and torch.isfinite(x).all() for x in masters)
    states = opt['optimizer_state_dict']['state']
    adam_steps, nonzero = [], 0
    for state in states.values():
        for value in state.values():
            if isinstance(value, torch.Tensor):
                assert torch.isfinite(value).all()
        adam_steps.append(int(state['step']))
        nonzero += bool(torch.count_nonzero(state['exp_avg']))
    assert adam_steps and min(adam_steps) > 0 and max(adam_steps) == step and nonzero > 0
    assert opt['lr_scheduler_type'] == 'constant'
    assert set(opt['scheduler_state_dict']['base_lrs']) == {3e-5}
    assert all(g['lr'] == 3e-5 for g in opt['optimizer_state_dict']['param_groups'])
    result = dict(created_utc=datetime.now(timezone.utc).isoformat(), verdict='PASS',
        checkpoint=str(path), initialization=str(INIT), global_step=step, groups=groups,
        optimizer=dict(master_tensors=len(masters), master_elements=sum(x.numel() for x in masters),
                       adam_step_min=min(adam_steps), adam_step_max=max(adam_steps),
                       nonzero_moments=nonzero, scheduler=opt['scheduler_state_dict']),
        scope='Actual optimizer updates and checkpoint numerical/frozen-backbone integrity only; not model-quality evidence')
    with out.open('x') as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
