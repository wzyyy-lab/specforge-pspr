#!/usr/bin/env python3
"""Falsification test for "the code/chat deficit is a training-data coverage problem".

The transfer-gap measurement compared benchmark-CODE and benchmark-CHAT slots against the
*pooled* training corpus.  That is not the comparison the data hypothesis needs.  PerfectBlend
is itself a blend -- it already contains code and chat -- so if its OWN code and chat rows also
sit at ~72% alt_rank_0 while its maths rows sit at ~81%, then there is no transfer gap at all in
those domains: the apparent deficit is the domain composition of the REFERENCE level, and adding
more code/chat data cannot lift a conditional that is already at its trained value.

So: apply ONE labelling function to BOTH corpora and compare like with like.

  * the label comes from the text of the supervised span (loss_mask == 1), decoded with the same
    tokenizer on both sides, so no format asymmetry can create the contrast;
  * the comparison is restricted to the reachable, repairable slots, exactly as before;
  * it is reported raw AND standardized onto the benchmark class's (slot x hill x draft-rank)
    cell mix, with the hill bins fitted once on the training corpus.

Reading:
  train_code ~= bench_code and train_chat ~= bench_chat  =>  coverage hypothesis FALSIFIED,
      the deficit is domain-intrinsic difficulty and P1 (more code/chat data) is the wrong lever.
  train_code >> bench_code                                =>  a real within-domain shift remains.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_transfer_gap import cells, covariates, frontier  # noqa: E402


def reachable(data, population: str):
    """Which slots count.  The external review's strongest objection to the pooled comparison was
    that ``frontier`` defines reachability through the SELECTOR's own earlier picks, so a selector
    that fails differently by domain selects different contexts, and it showed the pooled gap's
    magnitude moves with this rule (-2.42 / -3.98 / -4.86 pp).  The same-class comparison applies
    the rule symmetrically on both sides, but "symmetric" is an argument, not a measurement, so the
    rule is a knob and all three are reported.
    """
    tr = data["target_rank"].astype(np.int64)
    if population == "policy":
        return frontier(data)
    if population == "all":
        return tr >= 0
    if population == "base":
        # Reachable under the BASE walk: every earlier slot in the block had the truth at rank 0.
        slot = data["slot"].astype(np.int64)
        sample = data["sample"].astype(np.int64)
        block = data["block"].astype(np.int64)
        order = np.lexsort((slot, block, sample))
        key = sample[order] * (block.max() + 1) + block[order]
        new_group = np.empty(len(key), dtype=bool)
        new_group[0] = True
        new_group[1:] = key[1:] != key[:-1]
        fail = (tr[order] != 0).astype(np.int64)
        csum = np.cumsum(fail)
        gs = np.maximum.accumulate(np.where(new_group, np.arange(len(key)), 0))
        prior = csum - fail - np.where(gs > 0, csum[gs - 1], 0)
        out = np.zeros(len(key), dtype=bool)
        out[order] = prior == 0
        return out & (tr >= 0)
    raise ValueError(f"unknown population {population!r}")

NEEDED = ("lp", "scores", "target_rank", "slot", "sample", "block", "pick_index")

CODE_FENCE = re.compile(
    r"```[ \t]*(python|py|java|cpp|c\+\+|csharp|cs|c|javascript|js|typescript|ts|go|golang|rust|"
    r"ruby|rb|php|sql|bash|sh|shell|html|css|scala|kotlin|swift|matlab|perl|lua|haskell|r)\b",
    re.IGNORECASE,
)
CODE_BODY = re.compile(
    r"(\bdef [A-Za-z_]\w*\s*\(|\bclass [A-Za-z_]\w*\s*[:\(]|#include\s*[<\"]|"
    r"\bpublic\s+static\s+void\b|\bfunction\s+[A-Za-z_]\w*\s*\(|\bSELECT\b[\s\S]{0,200}?\bFROM\b|"
    r"\bconsole\.log\(|\bstd::|\bprintf\s*\()"
)
MATH_MARK = re.compile(r"(\\boxed|\\frac|\\times|\\sum|\\sqrt|\\begin\{|\\\[|####|\$\$)")
MATH_SOFT = re.compile(r"\$[^$\n]{1,60}\$")


def classify(text: str) -> str:
    """High-precision three-way label; ambiguous rows go to ``other`` and are reported apart."""
    if CODE_FENCE.search(text) or CODE_BODY.search(text):
        return "code"
    if MATH_MARK.search(text) or len(MATH_SOFT.findall(text)) >= 3:
        return "math"
    return "chat"


def load_features(path: str):
    with np.load(path) as d:
        out = {k: d[k] for k in NEEDED}
        files = [str(f) for f in d["files"]]
    return out, files


def doc_labels(files, tokenizer, cache_path: Path | None):
    if cache_path is not None and cache_path.exists():
        cached = json.loads(cache_path.read_text())
        if cached.get("files") == files:
            print(f"  (labels from cache {cache_path})")
            return cached["labels"]
    labels = []
    for i, f in enumerate(files):
        blob = torch.load(f, map_location="cpu", weights_only=False)
        ids = blob["input_ids"]
        mask = blob["loss_mask"].bool()
        text = tokenizer.decode(ids[mask].tolist(), skip_special_tokens=True)
        labels.append(classify(text))
        del blob
        if (i + 1) % 80 == 0:
            print(f"    labelled {i + 1}/{len(files)}", flush=True)
    if cache_path is not None:
        cache_path.write_text(json.dumps({"files": files, "labels": labels}))
    return labels


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-features", required=True)
    ap.add_argument("--eval-features", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--min-cell", type=int, default=40)
    ap.add_argument("--population", default="policy",
                    choices=("policy", "base", "all"),
                    help="which slots count as reachable; see reachable()")
    ap.add_argument("--cache-dir", default="outputs")
    ap.add_argument("--json-output", default=None)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    tr_d, tr_f = load_features(args.train_features)
    ev_d, ev_f = load_features(args.eval_features)
    cdir = Path(args.cache_dir)
    print("labelling training documents ...", flush=True)
    tr_lab = np.array(doc_labels(tr_f, tok, cdir / "DOMLAB_train.json"))
    print("labelling benchmark documents ...", flush=True)
    ev_lab = np.array(doc_labels(ev_f, tok, cdir / "DOMLAB_eval.json"))

    print("\n=== document counts under the shared labelling function ===")
    print(f"  {'class':<8} {'train docs':>11} {'bench docs':>11}")
    for c in ("math", "code", "chat"):
        print(f"  {c:<8} {int((tr_lab == c).sum()):>11} {int((ev_lab == c).sum()):>11}")

    # Cross-check the shared labels against the benchmark's known dataset provenance: if the text
    # classifier disagrees wildly with the dataset names, the labels are not measuring domain.
    dom_re = re.compile(r"([a-zA-Z0-9\-]+)_\d+\.ckpt$")
    ds = np.array([dom_re.search(Path(f).name).group(1) if dom_re.search(Path(f).name)
                   else "?" for f in ev_f])
    print("\n=== agreement of the text label with the benchmark dataset name ===")
    for d in sorted(set(ds)):
        k = ds == d
        tally = {c: int((ev_lab[k] == c).sum()) for c in ("math", "code", "chat")}
        print(f"  {d:<12} n={int(k.sum()):>4}  " +
              "  ".join(f"{c}={tally[c]}" for c in ("math", "code", "chat")))

    tr_reach, ev_reach = reachable(tr_d, args.population), reachable(ev_d, args.population)
    print(f"\n=== population = {args.population} ===")
    cov_t, edges = covariates(tr_d)
    cov_e, _ = covariates(ev_d, edges)
    mt = tr_reach & cov_t["repairable"]
    me = ev_reach & cov_e["repairable"]
    ct, ce = cells(cov_t, mt), cells(cov_e, me)
    ok_t, ok_e = cov_t["alt_ok"][mt], cov_e["alt_ok"][me]
    lab_t = tr_lab[tr_d["sample"].astype(np.int64)][mt]
    lab_e = ev_lab[ev_d["sample"].astype(np.int64)][me]

    lp_t, lp_e = tr_d["lp"].astype(np.float32), ev_d["lp"].astype(np.float32)
    rt, re_ = tr_d["target_rank"].astype(np.int64), ev_d["target_rank"].astype(np.int64)
    d1_t = ((rt > 0) & (lp_t[:, 1:].argmax(1) + 1 == rt))[mt]
    d1_e = ((re_ > 0) & (lp_e[:, 1:].argmax(1) + 1 == re_))[me]

    print("\n=== SAME-CLASS comparison, reachable repairable slots ===")
    print("  the reference level is now the training corpus's OWN rows of that class, not the pool.")
    print(f"  {'class':<8} {'n_train':>8} {'n_bench':>8} {'tr_d1':>7} {'be_d1':>7} "
          f"{'tr_a0':>7} {'be_a0':>7} {'raw_gap':>8} {'std_tr':>7} {'std_be':>7} "
          f"{'std_gap':>8} {'cover':>6}")
    out = {}
    for c in ("math", "code", "chat"):
        a, b = lab_t == c, lab_e == c
        if int(a.sum()) < 500 or int(b.sum()) < 500:
            print(f"  {c:<8} too few slots (train {int(a.sum())}, bench {int(b.sum())})")
            continue
        ta, ba = float(ok_t[a].mean()), float(ok_e[b].mean())
        ca, cb = ct[a], ce[b]
        w, num_t, num_e = 0.0, 0.0, 0.0
        for cell in np.unique(cb):
            ia, ib = ca == cell, cb == cell
            if int(ia.sum()) < args.min_cell or int(ib.sum()) < args.min_cell:
                continue
            n = int(ib.sum())
            w += n
            num_t += n * float(ok_t[a][ia].mean())
            num_e += n * float(ok_e[b][ib].mean())
        st, se = num_t / max(w, 1e-9), num_e / max(w, 1e-9)
        out[c] = dict(n_train=int(a.sum()), n_bench=int(b.sum()),
                      train_draft_r1=float(d1_t[a].mean()), bench_draft_r1=float(d1_e[b].mean()),
                      train_alt_r0=ta, bench_alt_r0=ba, raw_gap=ba - ta,
                      std_train=st, std_bench=se, std_gap=se - st,
                      coverage=w / max(int(b.sum()), 1))
        print(f"  {c:<8} {int(a.sum()):>8} {int(b.sum()):>8} "
              f"{100 * float(d1_t[a].mean()):>6.2f}% {100 * float(d1_e[b].mean()):>6.2f}% "
              f"{100 * ta:>6.2f}% {100 * ba:>6.2f}% {100 * (ba - ta):>+7.2f}pp "
              f"{100 * st:>6.2f}% {100 * se:>6.2f}% {100 * (se - st):>+7.2f}pp "
              f"{100 * w / max(int(b.sum()), 1):>5.1f}%")

    print("\n=== for reference: the pooled comparison that motivated P1 ===")
    print(f"  train(pooled) {100 * float(ok_t.mean()):.2f}%   bench(pooled) "
          f"{100 * float(ok_e.mean()):.2f}%   gap {100 * float(ok_e.mean() - ok_t.mean()):+.2f}pp")
    # Split the pooled gap into the part explained by class composition and the part left inside
    # the classes, so it is visible how much of the "transfer gap" was the reference level's mix.
    mix_e = {c: float((lab_e == c).mean()) for c in ("math", "code", "chat")}
    rw = sum(mix_e[c] * float(ok_t[lab_t == c].mean()) for c in mix_e if int((lab_t == c).sum()))
    pooled_gap = float(ok_e.mean() - ok_t.mean())
    within = float(ok_e.mean()) - rw
    print(f"  train reweighted onto the benchmark class mix: {100 * rw:.2f}%")
    print(f"  => composition {100 * (rw - float(ok_t.mean())):+.2f}pp, "
          f"within-class {100 * within:+.2f}pp, total {100 * pooled_gap:+.2f}pp "
          f"({100 * (1 - within / pooled_gap):.0f}% of the pooled gap was class composition)")
    print("  training-corpus class mix over reachable repairable slots: " +
          "  ".join(f"{c} {100 * float((lab_t == c).mean()):.1f}%" for c in ("math", "code", "chat")))
    print("  benchmark class mix:                                      " +
          "  ".join(f"{c} {100 * float((lab_e == c).mean()):.1f}%" for c in ("math", "code", "chat")))

    if args.json_output:
        Path(args.json_output).write_text(json.dumps(
            {"per_class": out,
             "pooled": {"train": float(ok_t.mean()), "bench": float(ok_e.mean())},
             "mix": {"train": {c: float((lab_t == c).mean()) for c in ("math", "code", "chat")},
                     "bench": {c: float((lab_e == c).mean()) for c in ("math", "code", "chat")}},
             "docs": {"train": {c: int((tr_lab == c).sum()) for c in ("math", "code", "chat")},
                      "bench": {c: int((ev_lab == c).sum()) for c in ("math", "code", "chat")}},
             "args": vars(args)}, indent=2, default=float))
        print(f"\nwrote {args.json_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
