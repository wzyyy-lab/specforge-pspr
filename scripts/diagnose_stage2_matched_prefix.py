"""Read-only, symmetric on-policy conditional comparison of frozen S2 and Domino.

Native generation is NOT reimplemented: hooks observe the unchanged decoder's
draft pass and target verification. A shadow draft has its own cropped KV cache
but receives the exact same verified target features, anchor and positions.
Both heads are then evaluated on the driver's realized prefix. Only accepted
positions and the first rejection have valid ground truth; the suffix is never
reported. This is a conditional diagnostic, NOT counterfactual accept length.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, DynamicCache

ROOT = Path(__file__).resolve().parents[2]
SF = ROOT / "SpecForge"
TAPS = ROOT / "TAPS-SP"
sys.path.insert(0, str(TAPS))
sys.path.insert(1, str(SF))
from model import DFlashDraftModel
from scripts.decode_lattice import (build_input, decode, lattice_block_inputs,
                                    lattice_score_slot, load_selector_checkpoint,
                                    slot_err_logit)
from scripts.domino_proposal_cap import install_proposal_cap
from specforge.modeling.draft.pspr_cloze import ClozeCorrector

EVAL = SF / "outputs/STAGE2_DOMINO_H15_EVAL200_20260907"
S2 = SF / "outputs/qwen3-4b-pspr-slotdeep-stage2-20260906_step9052_decode"
DOM = TAPS / "models/Qwen3-4B-Domino-b16"
TARGET = TAPS / "models/Qwen3-4B"
DOMAINS = ("gsm8k", "math500", "humaneval", "mbpp", "alpaca", "mt-bench")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def gate(scores, rho=3.0):
    # Exactly the native latgate comparison, including tie convention.
    prob = scores.softmax(-1)
    val, idx = prob[1:].max(-1)
    return int(idx) + 1 if float(val) > rho * float(prob[0]) else 0


def ranks(ids, truth):
    hit = (ids == truth).nonzero().flatten()
    return int(hit[0]) if hit.numel() else -1


class MatchedObserver:
    def __init__(self, target, s2, dom, selector, embed, driver, writer, prompt):
        self.target, self.s2, self.dom = target, s2, dom
        self.selector, self.embed = selector, embed
        self.driver, self.writer, self.prompt = driver, writer, prompt
        self.shadow_cache = DynamicCache()
        self.pending = None
        self.block = 0
        self.accepts = []
        self.parity_slots = 0
        self.handles = []

    def __enter__(self):
        native = self.s2 if self.driver == "stage2" else self.dom
        self.handles.append(native.register_forward_hook(self.on_draft, with_kwargs=True))
        self.handles.append(self.target.register_forward_hook(self.on_target, with_kwargs=True))
        return self

    def __exit__(self, *exc):
        for handle in self.handles:
            handle.remove()

    def on_draft(self, module, args, kwargs, output):
        assert self.pending is None, "draft called before previous verification"
        assert kwargs["noise_embedding"].shape[1] == 16
        start = int(kwargs["position_ids"][0, -16])
        context_len = kwargs["target_hidden"].shape[1]
        self.shadow_cache.crop(start - context_len)
        assert self.shadow_cache.get_seq_length() == start - context_len
        shadow = self.dom if self.driver == "stage2" else self.s2
        other = shadow(
            target_hidden=kwargs["target_hidden"],
            noise_embedding=kwargs["noise_embedding"],
            position_ids=kwargs["position_ids"],
            past_key_values=self.shadow_cache, use_cache=True, is_causal=False,
        )
        self.shadow_cache.crop(start)
        s2_raw, dom_raw = (output, other) if self.driver == "stage2" else (other, output)
        # These slices differ by design: PSPR excludes the anchor hidden, whereas
        # shifted Domino h[0] already predicts proposal 0. Never use [-15:] on DOM.
        sh = s2_raw[:, -15:, :]
        dh = dom_raw[:, :15, :]
        sb = self.target.lm_head(sh)[0].float()
        # Native Domino computes this matmul for all 16 hidden positions first.
        db = self.target.lm_head(dom_raw)[0, :15]
        self.pending = dict(start=start, sh=sh, dh=dh, sb=sb, db=db,
                            target_hidden=kwargs["target_hidden"])

    def on_target(self, module, args, kwargs, out):
        if self.pending is None:
            return  # Prefill, before any draft block.
        p = self.pending
        verify = args[0] if args else kwargs["input_ids"]
        assert verify.shape[1] == 16
        assert int(kwargs["position_ids"][0, 0]) == p["start"]
        truth = out.logits[0, :-1].argmax(-1)
        accept = int((verify[0, 1:] == truth).cumprod(0).sum())
        reachable = min(accept + 1, 15)
        self.accepts.append(accept + 1)
        sel = self.selector
        cand, lp, scal, cemb, z = lattice_block_inputs(
            sel, p["sb"], self.embed, p["sh"], p["target_hidden"], verify[0, :1])
        dcand = p["db"].float().topk(16, -1).indices
        # Correct top-1 tie convention is explicit; coverage is plain topk for
        # Domino, which itself is NOT restricted to this diagnostic set.
        dbase = p["db"].argmax(-1)
        sg = None
        _, dg = self.dom.prefix_gru(self.target.model.embed_tokens(verify[:, :2]))
        native_predictions = []
        decisions = []
        for i in range(15):
            prev = verify[0, i:i + 1]
            state, sg = sel.gru(self.embed[prev].unsqueeze(0), sg)
            score = lattice_score_slot(sel, z[i], state[0, 0], cemb[i], lp[i],
                                       cand[i], self.embed[prev][0],
                                       p["sh"][0, i].float(), prev[0], slot=i)
            si = gate(score)
            sp = int(cand[i, si])
            if i == 0:
                dp = int(dbase[i])  # Official pure_draft_prefix_len == 1.
            else:
                bias = self.dom.embed_proj(torch.cat(
                    [p["dh"][:, i:i + 1], dg.transpose(0, 1)], -1))
                # Preserve the checkpoint's BF16 addition, not a FP32 substitute.
                dp = int((p["db"][i].view(1, 1, -1) + bias).argmax(-1))
                if i + 1 < 15:
                    _, dg = self.dom.prefix_gru(
                        self.target.model.embed_tokens(verify[:, i + 1:i + 2]), dg)
            native_predictions.append(sp if self.driver == "stage2" else dp)
            if i >= reachable:
                continue  # Target labels beyond first rejection are invalid.
            tid = int(truth[i])
            sr = ranks(cand[i], tid)
            zero_score = lattice_score_slot(sel, z[i], torch.zeros_like(state[0, 0]),
                cemb[i], lp[i], cand[i], self.embed[prev][0], p["sh"][0, i].float(),
                prev[0], slot=i)
            delta_score = ClozeCorrector.score(sel, z[i], p["sh"][0, i].float(),
                                               cemb[i], lp[i], state[0, 0])
            err = slot_err_logit(sel, z, state, lp, scal, cemb, p["sh"], i, scores=score)
            tlogits = out.logits[0, i].float()
            ttop = tlogits.topk(2).values
            r = dict(slot=i, truth=tid, frontier=(i == accept),
                stage2_base=int(cand[i, 0]), stage2_pick=sp,
                stage2_truth_rank=sr, stage2_best=int(cand[i, score.argmax()]),
                stage2_rho1=int(cand[i, gate(score, 1.0)]),
                stage2_zero_state=int(cand[i, gate(zero_score)]),
                stage2_delta_only=int(cand[i, gate(delta_score)]),
                stage2_p_wrong=float(err.sigmoid()),
                stage2_best_alt=int(cand[i, int(score[1:].argmax()) + 1]),
                stage2_alt_margin=float(score[1:].max() - score[0]),
                stage2_truth_margin=float(score[sr] - score[0]) if sr >= 0 else None,
                stage2_truth_score_rank=int((score > score[sr]).sum()) if sr >= 0 else -1,
                domino_base=int(dbase[i]), domino_pick=dp,
                domino_truth_rank=ranks(dcand[i], tid),
                domino_pick_in_own_top16=bool((dcand[i] == dp).any()),
                target_top1_probability=float((ttop[0] - tlogits.logsumexp(0)).exp()),
                target_top1_margin=float(ttop[0] - ttop[1]))
            decisions.append(r)
        # Covers ALL proposals, including the unreported suffix. This checks
        # same-prefix conditional scoring reproduces the actual driver's choices.
        assert native_predictions == verify[0, 1:].tolist(), (
            self.driver, self.prompt["prompt_index"], self.block,
            native_predictions, verify[0, 1:].tolist())
        self.parity_slots += 15
        row = dict(dataset=self.prompt["dataset"], prompt_index=self.prompt["prompt_index"],
                   prompt_sha256=self.prompt["prompt_sha256"], driver=self.driver,
                   block=self.block, start=p["start"], anchor=int(verify[0, 0]),
                   acceptance_length=accept + 1, decisions=decisions)
        self.writer.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.writer.flush()
        self.block += 1
        self.pending = None


def generate(driver, target, s2, dom, sel, embed, inp, eos, max_new):
    if driver == "stage2":
        return decode("latgate", s2, target, None, sel, None, embed, inp,
            s2.mask_token_id, max_new, 16, [eos], gate_rho=3.0,
            gate_tau=0.0, gate_theta=0.0, gate_skip0=False, stop_policy="official")
    out = dom.spec_generate(target=target, input_ids=inp, max_new_tokens=max_new,
        block_size=16, stop_token_ids=[eos], temperature=0.0,
        graph_runner=None, use_bias=True, return_dict=True)
    return [int(x) for x in out.acceptance_lengths]


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", choices=DOMAINS, required=True)
    ap.add_argument("--max-samples", type=int, default=20)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--witness", action="store_true")
    args = ap.parse_args()
    assert args.max_samples > 0 and args.max_new_tokens > 0
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest = json.loads((EVAL / "prompts.json").read_text())
    prompts = [p for p in manifest["prompts"] if p["dataset"] == args.dataset][:args.max_samples]
    assert len(prompts) == args.max_samples
    assert all(hashlib.sha256(p["content"].encode()).hexdigest() == p["prompt_sha256"] for p in prompts)
    weights = [S2 / "backbone/model.safetensors", S2 / "selector.pt", DOM / "model.safetensors"]
    code = [Path(__file__), TAPS / "scripts/decode_lattice.py", TAPS / "scripts/domino_proposal_cap.py",
            DOM / "dflash.py", SF / "specforge/modeling/draft/pspr_slotdeep.py",
            SF / "specforge/modeling/draft/pspr_cloze.py"]
    hashes = {str(p): sha(p) for p in weights + code}
    args.output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(20260907)
    torch.set_num_threads(4)
    device = torch.device("cuda")
    target = AutoModelForCausalLM.from_pretrained(TARGET, attn_implementation="sdpa",
                dtype=torch.bfloat16).to(device).eval()
    s2 = DFlashDraftModel.from_pretrained(S2 / "backbone", attn_implementation="sdpa",
                dtype=torch.bfloat16).to(device).eval()
    dom = AutoModel.from_pretrained(DOM, trust_remote_code=True, attn_implementation="sdpa",
                dtype=torch.bfloat16).to(device).eval()
    cap = install_proposal_cap(dom, 15)
    sel, payload = load_selector_checkpoint(S2 / "selector.pt", device=device)
    sel = sel.float().eval()
    embed = target.model.embed_tokens.weight.float().detach()
    sel.bind_target_embedding(embed)
    assert type(sel).__name__ == "SlotDeepCorrector" and sel.K == 16
    assert s2.target_layer_ids == dom.target_layer_ids
    assert s2.block_size == dom.block_size == 16
    assert s2.mask_token_id == dom.mask_token_id and dom.pure_draft_prefix_len == 1
    assert dom.config.dflash_config["shift_label"]
    tokenizer = AutoTokenizer.from_pretrained(TARGET)
    original = json.loads((EVAL / "combined_results.json").read_text())["results"]
    lookup = {(r["prompt_sha256"], r["mode"]): r["acceptance_lengths"] for r in original}
    meta = dict(arguments={**vars(args), "output": str(args.output)}, hashes=hashes,
                manifest_sha256=sha(EVAL / "prompts.json"), cap=cap,
                decode_policy=payload.get("decode_policy"),
                device=torch.cuda.get_device_name(), target_dtype=str(target.dtype),
                selector_dtype=str(next(sel.parameters()).dtype),
                scope="conditional decisions at driver-reachable positions only; NOT oracle or counterfactual AL",
                prompt_hashes=[p["prompt_sha256"] for p in prompts])
    (args.output / "provenance.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n")
    results = []
    with (args.output / "blocks.jsonl").open("x") as stream:
        for prompt in prompts:
            inp = build_input(tokenizer, prompt["content"], device)
            for driver in ("stage2", "domino"):
                control = None
                if args.witness:
                    control = generate(driver, target, s2, dom, sel, embed, inp,
                                       tokenizer.eos_token_id, args.max_new_tokens)
                with MatchedObserver(target, s2, dom, sel, embed, driver, stream, prompt) as obs:
                    al = generate(driver, target, s2, dom, sel, embed, inp,
                                  tokenizer.eos_token_id, args.max_new_tokens)
                    assert obs.pending is None and al == obs.accepts
                assert al and all(1 <= a <= 16 for a in al)
                if control is not None:
                    assert control == al, ("hook changed native generation", driver, control, al)
                mode = "stage2_full" if driver == "stage2" else "domino_h15"
                reference = lookup.get((prompt["prompt_sha256"], mode))
                if args.max_new_tokens == 256:
                    assert reference is not None, ("missing original reference", mode)
                    assert reference == al, ("original eval200 parity failed", driver, prompt["prompt_index"], reference, al)
                results.append(dict(dataset=prompt["dataset"], prompt_index=prompt["prompt_index"],
                    driver=driver, prompt_sha256=prompt["prompt_sha256"], acceptance_lengths=al,
                    native_decision_parity_slots=obs.parity_slots,
                    control_parity=control is not None, eval200_parity=args.max_new_tokens == 256))
                print(json.dumps(results[-1]), flush=True)
    assert hashes == {str(p): sha(p) for p in weights + code}, "input changed during diagnostic"
    done = dict(results=results, blocks_sha256=sha(args.output / "blocks.jsonl"),
                input_hashes_unchanged=True, status="complete")
    (args.output / "completion.json").write_text(json.dumps(done, indent=2) + "\n")
    print("MATCHED_PREFIX_WITNESS_PASS" if args.witness else "MATCHED_PREFIX_DIAGNOSTIC_PASS", flush=True)


if __name__ == "__main__":
    main()
