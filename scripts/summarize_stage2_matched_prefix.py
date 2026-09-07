"""Summarize raw matched-state decisions. No counterfactual AL estimator."""
import argparse
import hashlib
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path


DOMAINS = ("gsm8k", "math500", "humaneval", "mbpp", "alpaca", "mt-bench")
VARIANTS = ("stage2_pick", "stage2_rho1", "stage2_zero_state", "stage2_delta_only")


def auc(pairs):
    """Exact rank AUC with half credit for equal scores."""
    grouped = defaultdict(lambda: [0, 0])
    for score, positive in pairs:
        grouped[score][int(positive)] += 1
    positives = sum(n[1] for n in grouped.values())
    negatives = sum(n[0] for n in grouped.values())
    if not positives or not negatives:
        return None
    less_negative, wins = 0, 0.0
    for score in sorted(grouped):
        n0, n1 = grouped[score]
        wins += n1 * (less_negative + .5 * n0)
        less_negative += n0
    return wins / (positives * negatives)


def summarize(blocks):
    cnt = Counter(blocks=len(blocks), accepted_sum=sum(b["acceptance_length"] for b in blocks))
    variants = {v: Counter() for v in VARIANTS}
    slot_bins = defaultdict(Counter)
    errors = []
    margins = defaultdict(list)
    for block in blocks:
        ds = block["decisions"]
        assert len(ds) == min(block["acceptance_length"], 15)
        assert [r["slot"] for r in ds] == list(range(len(ds)))
        assert sum(r["frontier"] for r in ds) == int(block["acceptance_length"] < 16)
        for r in ds:
            t = r["truth"]
            sb, sp = r["stage2_base"] == t, r["stage2_pick"] == t
            db, dp = r["domino_base"] == t, r["domino_pick"] == t
            sc, dc = r["stage2_truth_rank"] >= 0, r["domino_truth_rank"] >= 0
            native_ok = sp if block["driver"] == "stage2" else dp
            assert native_ok != r["frontier"]
            cnt.update(n=1, frontier=int(r["frontier"]), stage2_base_correct=int(sb),
                stage2_correct=int(sp), domino_base_correct=int(db), domino_correct=int(dp),
                stage2_covered=int(sc), domino_covered=int(dc),
                both_correct=int(sp and dp), both_wrong=int(not sp and not dp),
                domino_only_correct=int(dp and not sp), stage2_only_correct=int(sp and not dp))
            for model, base, pick, cov in (("stage2", sb, sp, sc), ("domino", db, dp, dc)):
                cnt[f"{model}_base_wrong"] += not base
                cnt[f"{model}_repairable"] += cov and not base
                cnt[f"{model}_repaired"] += not base and pick
                cnt[f"{model}_repaired_in_top16"] += not base and pick and cov
                cnt[f"{model}_destroyed"] += base and not pick
            if dp and not db:
                cnt["domino_repair_outside_own_top16"] += not dc
            if dp and not sp:
                cnt["domino_win_base_already_correct"] += db
                cnt["domino_win_head_required"] += not db
                cnt["domino_win_stage2_outside_top16"] += not sc
                if not sc:
                    kind = "outside_top16"
                elif sb:
                    kind = "destroyed_base_correct"
                elif r["stage2_best"] == t:
                    kind = "truth_best_gate_blocked"
                else:
                    kind = "rank_failure"
                cnt["domino_win_" + kind] += 1
            if r["frontier"] and block["driver"] == "stage2":
                if not sc:
                    kind = "outside_top16"
                elif sb:
                    kind = "destroyed_base_correct"
                elif r["stage2_best"] == t:
                    kind = "truth_best_gate_blocked"
                else:
                    kind = "rank_failure"
                cnt["stage2_frontier_" + kind] += 1
            for v, vc in variants.items():
                vp = r[v] == t
                vc.update(correct=int(vp), repaired=int(not sb and vp),
                    destroyed=int(sb and not vp),
                    gained_vs_native=int(vp and not sp), lost_vs_native=int(sp and not vp))
            bins = "1" if r["slot"] == 0 else "2-4" if r["slot"] < 4 else "5-8" if r["slot"] < 8 else "9-15"
            slot_bins[bins].update(n=1, stage2_base_correct=int(sb), domino_base_correct=int(db),
                                 stage2_correct=int(sp), domino_correct=int(dp),
                                 stage2_covered=int(sc), domino_covered=int(dc))
            errors.append((r["stage2_p_wrong"], not sb))
            if dp and not sp:
                margins["domino_only_correct"].append(r["target_top1_margin"])
            if not dp and not sp:
                margins["both_wrong"].append(r["target_top1_margin"])
    for key in ("outside_top16", "destroyed_base_correct", "truth_best_gate_blocked", "rank_failure"):
        cnt.setdefault("domino_win_" + key, 0)
    assert sum(cnt["domino_win_" + k] for k in
        ("outside_top16", "destroyed_base_correct", "truth_best_gate_blocked", "rank_failure")) == cnt["domino_only_correct"]
    return dict(counts=dict(cnt), variants={k: dict(v) for k, v in variants.items()},
                slot_bins={k: dict(v) for k, v in slot_bins.items()},
                error_detector_auc=auc(errors),
                target_margin={k: dict(n=len(v), mean=statistics.mean(v), median=statistics.median(v))
                               for k, v in margins.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("outputs/STAGE2_MATCHED_PREFIX_20260907"))
    args = ap.parse_args()
    output = args.root / "summary.json"
    if output.exists():
        raise FileExistsError(output)
    groups = defaultdict(list)
    inputs = {}
    total_parity = 0
    for ds in DOMAINS:
        p = args.root / ds / "blocks.jsonl"
        done = json.loads((p.parent / "completion.json").read_text())
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        assert done["status"] == "complete" and done["input_hashes_unchanged"]
        assert h == done["blocks_sha256"]
        assert len(done["results"]) == 40
        assert all(r["eval200_parity"] for r in done["results"])
        inputs[str(p)] = h
        blocks = [json.loads(line) for line in p.open()]
        for b in blocks:
            assert b["dataset"] == ds
            groups[(ds, b["driver"])].append(b)
        for row in done["results"]:
            subset = [b for b in groups[(ds, row["driver"])] if b["prompt_index"] == row["prompt_index"]]
            assert [b["acceptance_length"] for b in subset] == row["acceptance_lengths"]
            assert len(subset) * 15 == row["native_decision_parity_slots"]
            total_parity += row["native_decision_parity_slots"]
    summary = {ds: {driver: summarize(groups[(ds, driver)]) for driver in ("stage2", "domino")}
               for ds in DOMAINS}
    payload = dict(scope="20 fixed prompts per domain, two native drivers; conditional decisions, NOT counterfactual AL or heldout performance",
                   evaluation_type="synthetic_proxy", native_decision_parity_slots=total_parity,
                   inputs=inputs, results=summary)
    with output.open("x") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")
    print("MATCHED_PREFIX_SUMMARY_PASS", total_parity)
    print("domain driver n S2base% DOMbase% S2full% DOMfull% DOMwinN DOMwinBaseCorrect% DOMwinS2Miss%")
    for ds in DOMAINS:
        for driver in ("stage2", "domino"):
            c = summary[ds][driver]["counts"]
            n, w = c["n"], c["domino_only_correct"]
            print(ds, driver, n, *(round(100*c[k]/n, 3) for k in
                ("stage2_base_correct", "domino_base_correct", "stage2_correct", "domino_correct")),
                w, round(100*c["domino_win_base_already_correct"]/max(w, 1), 3),
                round(100*c["domino_win_stage2_outside_top16"]/max(w, 1), 3))


if __name__ == "__main__":
    main()
