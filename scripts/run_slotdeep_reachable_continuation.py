"""Generate, record and launch a bounded matched online SlotDeep continuation.

Uses the completed Stage2 resolved recipe, not stale example-YAML defaults.
The two arms differ only in selector CE/alternative-CE slot weighting.
All generated artifacts use exclusive paths; this never resumes/overwrites a run.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess

import yaml

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / 'outputs/SLOTDEEP_REACHABLE_20260907'
INIT = ROOT / ('outputs/qwen3-4b-pspr-slotdeep-stage2-20260906/'
               'qwen3-4b-pspr-slotdeep-stage2-20260906-step9052/training_state.pt')
ENV_SPEC = '/home/wangzhuoyu/pspr-sglang-0515-stage2-h20-spec.json'


def build_config(arm, phase):
    archived = ROOT / 'outputs/SLOTDEEP_stage2_20260906_PROVENANCE'
    cfg = json.loads((archived / 'resolved.json').read_text())
    draft = json.loads((archived / 'draft_config.json').read_text())
    draft['dflash_config']['freeze_backbone'] = True
    name = f'slotdeep-reachable-{arm}-{phase}-20260907'
    run = WORK / name
    steps = 20 if phase == 'smoke' else 1000
    cfg['run_id'], cfg['output_dir'] = name, str(run / 'checkpoints')
    cfg['model'].update(draft_checkpoint_path=str(INIT), draft_model_config=str(run / 'draft.json'))
    # Frozen *trained Stage2* backbone, not the official DFlash release.
    t = cfg['training']
    t.update(max_steps=steps, total_steps=None, num_epochs=1, batch_size=2,
             accumulation_steps=4, learning_rate=3e-5, lr_scheduler='constant',
             warmup_ratio=.05, lr_scale_rules={'candidate_selector.': 1.},
             save_interval=steps if phase == 'smoke' else 250,
             log_interval=5 if phase == 'smoke' else 10, resume_from=None,
             seed=42, prompt_seed=20260907,
             dflash2_selector_weight_mode=('uniform_frontier_boost' if arm == 'control'
                                            else 'reachable_frontier_boost'),
             dflash2_selector_survival_floor=.25)
    # Equal batch geometry: 3 trainers * 2 examples * 4 accumulation = 24 per arm.
    offset = 0 if arm == 'control' else 4
    cfg['deployment']['trainer'].update(nproc_per_node=3, master_port=29610 + offset)
    d = cfg['deployment']['disaggregated']
    d.update(control_dir=str(run/'control'), consumer_state_dir=str(run/'consumer-state'))
    m = d['managed_local']
    m['trainer_cuda_visible_devices'] = [str(offset + i) for i in (1, 2, 3)]
    m['capture_servers'][0].update(port=31000 + offset, cuda_visible_devices=[str(offset)])
    m['mooncake'].update(rpc_port=36551 + offset, metadata_port=36880 + offset,
                         metrics_port=36903 + offset)
    return cfg, draft, run


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arm', choices=['control', 'reachable'], required=True)
    p.add_argument('--phase', choices=['smoke', 'train'], required=True)
    p.add_argument('--plan', action='store_true', help='read-only, print exact resolved input and GPU assignment')
    args = p.parse_args()
    cfg, draft, run = build_config(args.arm, args.phase)
    if args.plan:
        print(json.dumps(dict(config=cfg, draft=draft, run=str(run)), indent=2))
        return
    if not INIT.is_file():
        raise FileNotFoundError(INIT)
    usage = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used',
                                    '--format=csv,noheader,nounits'], text=True)
    used = {int(a): int(b) for a, b in (line.split(',') for line in usage.splitlines())}
    offset = 0 if args.arm == 'control' else 4
    if any(used.get(i, 999999) >= 500 for i in range(offset, offset+4)):
        raise RuntimeError(f'Assigned GPUs not free: {used}')
    env = os.environ.copy()
    env.update(PYTHONPATH='/home/wangzhuoyu/sglang-patched-0.5.15:.',
               SLOTDEEP_ENV_SPEC=ENV_SPEC, SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK='1',
               OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
               HF_HUB_OFFLINE='1', no_proxy='localhost,127.0.0.1', NO_PROXY='localhost,127.0.0.1')
    env.pop('CUDA_VISIBLE_DEVICES', None)  # topology owns physical GPU assignment
    run.mkdir(parents=True, exist_ok=False)
    (run/'draft.json').write_text(json.dumps(draft, indent=2)+'\n')
    (run/'train.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
    subprocess.run(['python', 'scripts/record_slotdeep_launch.py', '--config', str(run/'train.yaml'),
                    '--out', str(run/'provenance')],
                   cwd=ROOT, env=env, check=True)
    command = ['python', '-u', '-m', 'specforge.cli', 'train', '-c', str(run/'train.yaml')]
    with (run/'train.log').open('x') as log:
        print('ONLINE_TRAIN_START', args.arm, args.phase, str(run), flush=True)
        proc = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    (run/'completion.json').write_text(json.dumps(dict(returncode=proc.returncode, command=command,
        ended_utc=datetime.now(timezone.utc).isoformat()), indent=2)+'\n')
    if proc.returncode:
        raise SystemExit(proc.returncode)
    print('ONLINE_TRAIN_DONE', args.arm, args.phase, str(run), flush=True)
    with (run/'verify.log').open('x') as log:
        subprocess.run(['python', 'scripts/verify_slotdeep_reachable_checkpoint.py',
                        '--arm', args.arm, '--phase', args.phase], cwd=ROOT,
                       env=dict(env, PYTHONPATH='.'), stdout=log,
                       stderr=subprocess.STDOUT, check=True)
    print('CHECKPOINT_VERIFIED', args.arm, args.phase, flush=True)
    if args.phase == 'train':
        # Preserve results across an SSH disconnect: only a successful training
        # exit can advance to verification/export and a fixed native-policy eval.
        eval_env = dict(env, PYTHONPATH='.', CUDA_VISIBLE_DEVICES=str(offset),
                        HF_DATASETS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
        checkpoint = Path(cfg['output_dir']) / f'{cfg["run_id"]}-step1000' / 'training_state.pt'
        backbone = ROOT/'outputs/qwen3-4b-pspr-slotdeep-stage2-20260906_step9052_decode/backbone'
        export = run/'step1000_decode'
        commands = [
            ('export', ['python', 'scripts/export_pspr_cloze_for_decode.py',
                        '--checkpoint', str(checkpoint), '--draft-config', str(run/'draft.json'),
                        '--backbone-template', str(backbone), '--verify-frozen', str(backbone),
                        '--verify-roundtrip', '--out', str(export)]),
            ('validation', ['python', '-u', 'scripts/evaluate_pspr_holdout.py',
                            '--prompts', str(ROOT/'cache/dataset/pspr_holdout_20260905/validation.jsonl'),
                            '--export', str(export), '--output', str(run/'pb_validation.json'),
                            '--max-new-tokens', '256', '--max-prompt-tokens', '2816', '--gate-stats']),
        ]
        for label, command in commands:
            print('POST_TRAIN_START', args.arm, label, flush=True)
            with (run/f'{label}.log').open('x') as log:
                subprocess.run(command, cwd=ROOT, env=eval_env, stdout=log,
                               stderr=subprocess.STDOUT, check=True)
        (run/'pipeline_completion.json').write_text(json.dumps(dict(returncode=0,
            ended_utc=datetime.now(timezone.utc).isoformat(),
            scope='Training, frozen-checkpoint verification, export, PB validation complete; no quality pass implied'),
            indent=2)+'\n')
        print('CONTINUATION_PIPELINE_DONE', args.arm, flush=True)


if __name__ == '__main__':
    main()
