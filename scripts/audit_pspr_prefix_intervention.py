"""Paired HF counterfactual: restore empty-think prefix for supervised history turns.

Uses all eligible spans in the fixed prior cache sample. Inserts only the current
assistant's missing generation-prefix tokens; earlier context and scored corpus
tokens are unchanged. Both variants run in one equally padded batch. This is a
teacher-label diagnostic, NOT measured acceptance or a training intervention.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import numpy as np
from scripts.audit_pspr_training_labels import sha


def make_pair(ids, mask, header_end, end, insertion):
    positions = [i for i in range(header_end, end) if mask[i] > 0]
    base = ids[:end]
    changed = ids[:header_end] + insertion + ids[header_end:end]
    shifted = [i + len(insertion) for i in positions]
    assert [base[i] for i in positions] == [changed[i] for i in shifted]
    return base, changed, positions, shifted


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--label-audit", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    original = json.loads(args.label_audit.read_text())
    from datasets import Dataset
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    target_path = original["arguments"]["target_model"]
    tok = AutoTokenizer.from_pretrained(target_path)
    target = AutoModelForCausalLM.from_pretrained(target_path, dtype=torch.bfloat16,
        attn_implementation="sdpa").to("cuda").eval()
    header = tok.encode("<|im_start|>assistant\n", add_special_tokens=False)
    insertion = tok.encode("<think>\n\n</think>\n\n", add_special_tokens=False)
    eos = tok.convert_tokens_to_ids("<|im_end|>")
    special = set(tok.all_special_ids)
    tables, results = {}, []
    totals = defaultdict(int)
    for sample in original["per_sample"]:
        file = sample["cache_file"]
        if file not in tables:
            assert sha(file) == original["source_sha256"][str(Path(file).resolve())]
            tables[file] = Dataset.from_file(file)
        row = tables[file][sample["cache_row"]]
        ids, mask = row["input_ids"][0], row["loss_mask"][0]
        assert hashlib.sha256(np.asarray(ids,dtype=np.int64).tobytes()).hexdigest() == sample["input_ids_sha256"]
        for start in range(len(ids)-len(header)+1):
            if ids[start:start+len(header)] != header:
                continue
            h_end = start+len(header)
            if ids[h_end:h_end+len(insertion)] == insertion:
                continue
            end = next((i for i in range(h_end,len(ids)) if ids[i] == eos), len(ids))
            base, changed, positions, shifted = make_pair(ids,mask,h_end,end,insertion)
            eligible = [(a,b) for a,b in zip(positions,shifted) if ids[a] not in special]
            if not eligible:
                continue
            pos0,pos1 = zip(*eligible)
            batch = torch.full((2,len(changed)),tok.pad_token_id,device="cuda",dtype=torch.long)
            attention = torch.zeros_like(batch)
            batch[0,:len(base)] = torch.tensor(base,device="cuda")
            batch[1] = torch.tensor(changed,device="cuda")
            attention[0,:len(base)] = 1
            attention[1] = 1
            ix0 = torch.tensor(pos0,device="cuda")-1
            ix1 = torch.tensor(pos1,device="cuda")-1
            with torch.inference_mode():
                logits = target(batch,attention_mask=attention,use_cache=False).logits
                assert torch.isfinite(logits).all()
                a = logits[0,ix0].argmax(-1)
                b = logits[1,ix1].argmax(-1)
                truth = batch[0,ix0+1]
                ok0,ok1 = a.eq(truth),b.eq(truth)
                counts = dict(tokens=truth.numel(), baseline_mismatches=int((~ok0).sum()),
                    corrected_mismatches=int((~ok1).sum()), fixed=int((~ok0 & ok1).sum()),
                    harmed=int((ok0 & ~ok1).sum()))
            assert counts["baseline_mismatches"]-counts["corrected_mismatches"] == counts["fixed"]-counts["harmed"]
            for k,v in counts.items():
                totals[k] += v
            results.append(dict(global_cache_row=sample["global_cache_row"],cache_file=file,
                cache_row=sample["cache_row"],header_start=start,content_end=end,**counts))
            del logits,batch,attention,ix0,ix1,a,b,truth,ok0,ok1
            if len(results)%16 == 0:
                print("PREFIX_PAIR",len(results),dict(totals),flush=True)
    assert results
    groups = defaultdict(lambda: np.zeros(3,dtype=np.float64))
    for row in results:
        groups[row["global_cache_row"]] += [row["tokens"],row["baseline_mismatches"],row["corrected_mismatches"]]
    values = np.stack(list(groups.values()))
    rng = np.random.default_rng(20260905)
    boot = values[rng.integers(0,len(values),(10000,len(values)))].sum(1)
    delta = (boot[:,2]-boot[:,1])/boot[:,0]*100
    result = dict(scope=__doc__,label_audit=str(args.label_audit),label_audit_sha256=sha(args.label_audit),
        source_sha256={str(Path(__file__).resolve()):sha(__file__),
                       str(Path("scripts/audit_pspr_training_labels.py").resolve()):sha("scripts/audit_pspr_training_labels.py")},
        paired_batch_size=2,paired_padding="right-padded with explicit attention mask",
        target_model=target_path,scored_population="nonspecial, supervised content tokens in all missing-empty-think spans of the prior fixed cache sample",
        totals=dict(totals),spans=len(results),original_rows=len(groups),
        paired_row_bootstrap_95ci_delta_mismatch_pp=np.quantile(delta,[.025,.975]).tolist(),per_span=results)
    with args.output.open("x") as sink:
        json.dump(result,sink,indent=2)
        sink.write("\n")
    print("PREFIX_PAIR_DONE",json.dumps({k:result[k] for k in ("totals","spans","original_rows","paired_row_bootstrap_95ci_delta_mismatch_pp")}),flush=True)


if __name__ == "__main__":
    main()
