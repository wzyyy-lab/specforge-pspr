"""Read-only input audit; writes only a new JSON witness, never training data."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from scripts.prepare_pspr_holdout import prompt_key, user_turns
from scripts.run_slotdeep_reachable_continuation import ROOT, INIT


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    train = ROOT/'cache/dataset/perfectblend_qwen3-4b_regen_turnwise_20260905.jsonl'
    meta = json.loads(train.with_suffix('.manifest.json').read_text())
    assert sha(train) == meta['output_sha256']
    source = ROOT/'cache/dataset/perfectblend_qwen3-4b_regen.jsonl'
    assert sha(source) == meta['source_sha256'][str(source)]
    ids, count = set(), 0
    with train.open() as f:
        for line in f:
            row = json.loads(line)
            ids.add(row['source_id'])
            assert len(row['input_ids']) == len(row['loss_mask']) <= 3072
            count += 1
    assert count == meta['counts']['kept_turns']
    splits = {}
    for label in ('calibration', 'validation'):
        path = ROOT/f'cache/dataset/pspr_holdout_20260905/{label}.jsonl'
        rows = [json.loads(line) for line in path.open()]
        keys = {prompt_key(row['turns'][0]) for row in rows}
        assert len(keys) == len(rows) == 512
        splits[label] = dict(path=str(path), sha256=sha(path), keys=keys, rows=len(rows), overlap=set())
    found, source_rows, turns = set(), 0, 0
    with source.open() as f:
        for line in f:
            row = json.loads(line)
            if row['id'] not in ids:
                continue
            found.add(row['id'])
            source_rows += 1
            for text in user_turns(row):
                key = prompt_key(text)
                turns += 1
                for split in splits.values():
                    if key in split['keys']:
                        split['overlap'].add(key)
    assert found == ids
    for split in splits.values():
        assert not split['overlap'], f'Holdout intersects current training source: {split["path"]}'
        split.pop('keys')
        split['overlap'] = sorted(split['overlap'])
    result = dict(created_utc=datetime.now(timezone.utc).isoformat(), status='PASS',
        training=dict(path=str(train), sha256=meta['output_sha256'], rows=count,
                      retained_source_ids=len(ids), source_rows=source_rows, source_user_turns=turns),
        initialization=str(INIT), splits=splits,
        scope='Exact normalized first-prompt exclusion against all user turns of retained regen source rows; input hash/lineage verified. Not semantic or pretraining decontamination. Historical benchmark overlap is not erased.')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
