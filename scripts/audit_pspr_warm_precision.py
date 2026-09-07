"""Materialize the legacy wrapper's FP32->BF16->FP32 round trip, never overwrite input."""
import argparse
import hashlib
import json
from pathlib import Path
import torch

p=argparse.ArgumentParser()
p.add_argument('--source',required=True)
p.add_argument('--out',required=True)
a=p.parse_args()
source, out=Path(a.source),Path(a.out)
if out.exists() or source.resolve()==out.resolve():
    raise ValueError('diagnostic output must be a new path')
payload=torch.load(source,map_location='cpu',weights_only=True)
details={}
for name,value in payload['state_dict'].items():
    if value.dtype==torch.float32:
        rounded=value.bfloat16().float()
        details[name]=dict(n=value.numel(), changed=int((value!=rounded).sum()),
                           max_abs=float((value-rounded).abs().max()))
        payload['state_dict'][name]=rounded
payload['precision_audit']=dict(source=str(source),
    source_sha256=hashlib.file_digest(source.open('rb'),'sha256').hexdigest(),
    operation='legacy wrapper fp32 -> bf16 -> fp32, no optimization',tensors=details)
out.parent.mkdir(parents=True,exist_ok=True)
torch.save(payload,out)
out.with_suffix('.audit.json').write_text(json.dumps(payload['precision_audit'],indent=2)+'\n')
print('changed',sum(v['changed'] for v in details.values()),'/',sum(v['n'] for v in details.values()))
print(out)
