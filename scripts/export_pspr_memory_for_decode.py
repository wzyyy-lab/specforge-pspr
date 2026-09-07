"""Independent fail-closed PSPR-Memory exporter and step-zero warm-start witness."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import torch
from safetensors.torch import load_file, save_file
from specforge.modeling.draft.pspr_memory import MemoryCorrector


def reference_config(config):
    d = config['dflash_config']
    return dict(hidden_dim=config['hidden_size'], vocab_size=config['vocab_size'],
                K=d['selector_top_k'], d=d['selector_dim'], n_layers=d['selector_layers'],
                n_heads=d['selector_heads'], ds=d['selector_state_dim'],
                delta_h=d['selector_delta_hidden'], max_slots=d['selector_max_slots'],
                dropout=d['selector_dropout'], direct_hidden=d['selector_direct_hidden'],
                use_state=d['selector_use_state'], bidirectional=d['selector_bidirectional'],
                memory_length=d['memory_length'], memory_source=d['memory_source'],
                memory_future_causal=d.get('memory_future_causal', False),
                memory_dropout=d['memory_dropout'])


def main():
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--checkpoint')
    g.add_argument('--seed-selector')
    p.add_argument('--draft-config', required=True)
    p.add_argument('--backbone-template', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--verify-frozen', action='store_true')
    args = p.parse_args()
    config = json.loads(Path(args.draft_config).read_text())
    torch.manual_seed(42)
    head = MemoryCorrector.from_reference_config(reference_config(config)).eval()
    ref = load_file(str(Path(args.backbone_template) / 'model.safetensors'))
    objective, step = 'multiclass', 0
    if args.seed_selector:
        source = torch.load(args.seed_selector, map_location='cpu', weights_only=True)
        if source.get('model_type') != 'ClozeCorrector':
            raise ValueError('step-zero warm start must be a ClozeCorrector artifact')
        result = head.load_state_dict(source['state_dict'], strict=False)
        assert not result.unexpected_keys
        assert result.missing_keys and all(k.startswith('memory_refiner.') for k in result.missing_keys)
        assert torch.count_nonzero(head.memory_refiner.out.weight) == 0
        backbone = ref
    else:
        payload = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
        state = payload['draft_state_dict']
        head.load_state_dict({k.removeprefix('candidate_selector.'): v for k, v in state.items()
                              if k.startswith('candidate_selector.')}, strict=True)
        backbone = {k: v for k, v in state.items() if not k.startswith('candidate_selector.')}
        step = int(payload.get('global_step', -1))
        objective = payload.get('dflash2_selector_objective')
        if step < 0 or objective != 'multiclass':
            raise ValueError(f'unsupported/missing training provenance: {step=}, {objective=}')
    if args.verify_frozen:
        assert set(backbone) == set(ref), 'backbone keys differ'
        assert all(torch.equal(backbone[k].float(), ref[k].float()) for k in ref), 'backbone changed'
        print(f'FROZEN_BACKBONE_EXACT {len(ref)} tensors')
    d = config['dflash_config']
    policy = dict(selector_decision_mode=d['selector_decision_mode'],
                  selector_compute_dtype=d['selector_compute_dtype'], selector_top_k=d['selector_top_k'],
                  selector_gate_rho=d['selector_gate_rho'], selector_gate_tau=d['selector_gate_tau'],
                  selector_gate_theta=d['selector_gate_theta'], selector_gate_skip_first=False,
                  selector_keep_repair_margin=0.0)
    if policy['selector_decision_mode'] != 'margin_gate':
        raise ValueError('unvalidated memory serving policy')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)  # never overwrite an existing run/export
    bb = out / 'backbone'
    bb.mkdir()
    for filename in ('config.json', 'dflash.py', 'modeling_dflash.py', 'utils.py'):
        path = Path(args.backbone_template) / filename
        if path.exists():
            shutil.copy2(path, bb / filename)
    save_file({k: v.contiguous() for k, v in backbone.items()}, str(bb / 'model.safetensors'))
    artifact = dict(model_type='MemoryCorrector', config=head.reference_config(),
                    state_dict=head.state_dict(), global_step=step, decode_policy=policy,
                    training_policy=dict(verified=True, selector_objective=objective,
                                         backbone='frozen' if args.verify_frozen else 'trained', **policy),
                    provenance=dict(checkpoint=args.checkpoint, seed_selector=args.seed_selector,
                                    draft_config=args.draft_config))
    torch.save(artifact, out / 'selector.pt')
    reloaded = torch.load(out / 'selector.pt', map_location='cpu', weights_only=True)
    check = MemoryCorrector.from_reference_config(reloaded['config'])
    check.load_state_dict(reloaded['state_dict'], strict=True)
    assert all(torch.equal(v, check.state_dict()[k]) for k, v in head.state_dict().items())
    stats = dict(total_parameters=sum(p.numel() for p in head.parameters()),
                 residual_parameters=sum(p.numel() for p in head.memory_refiner.parameters()),
                 global_step=step, roundtrip_exact=True, **head.reference_config())
    (out / 'VERIFIED.json').write_text(json.dumps(stats, indent=2) + '\n')
    print(json.dumps(stats, indent=2))


if __name__ == '__main__':
    main()
