"""Fixed rho3-vs-rho1 decision-band analysis, not threshold fitting.

The band contains exactly decisions changed by rho1 on the same realized
prefix. KEEP-correct, best-alternative-correct and neither are disjoint. AUC
compares repair vs damage, explicitly excluding neither. Bootstrap resamples
whole prompts, not tokens, and measures prompt-composition sensitivity only.
"""
import argparse
import hashlib
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

from summarize_stage2_matched_prefix import DOMAINS, auc


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path("outputs/STAGE2_MATCHED_PREFIX_20260907"))
    ap.add_argument("--bootstrap", type=int, default=2000)
    args = ap.parse_args()
    path = args.root / "decision_boundary.json"
    if path.exists():
        raise FileExistsError(path)
    out = dict(scope="Fixed-state inference diagnostics, NOT counterfactual accept length, no threshold fitted",
               driver="stage2", seed=20260907, bootstrap=args.bootstrap,
               auc_population="best-alternative-correct vs KEEP-correct within rho3-to-rho1 changed-action band; neither excluded",
               ci_scope="20 development prompts/domain, cluster bootstrap prompt-composition sensitivity only",
               domains={})
    for domain in DOMAINS:
        p = args.root / domain / "blocks.jsonl"
        done = json.loads((p.parent / "completion.json").read_text())
        assert done["status"] == "complete"
        assert hashlib.sha256(p.read_bytes()).hexdigest() == done["blocks_sha256"]
        rows = [(b["prompt_index"], r) for line in p.open() for b in [json.loads(line)]
                if b["driver"] == "stage2" for r in b["decisions"]]
        by_prompt = defaultdict(list)
        for pid, row in rows:
            by_prompt[pid].append(row)
        assert len(by_prompt) == 20
        all_rows = [r for _, r in rows]
        band = [r for r in all_rows if r["stage2_pick"] != r["stage2_rho1"]]
        good = [r for r in band if r["stage2_rho1"] == r["truth"]]
        bad = [r for r in band if r["stage2_base"] == r["truth"]]
        assert all(r["stage2_pick"] == r["stage2_base"] for r in band)
        pairs = [(r["stage2_p_wrong"], True) for r in good] + [(r["stage2_p_wrong"], False) for r in bad]
        grouped = {}
        for pid, rs in by_prompt.items():
            grouped[pid] = [(r["stage2_p_wrong"], r["stage2_rho1"] == r["truth"])
                for r in rs if r["stage2_pick"] != r["stage2_rho1"] and
                (r["stage2_rho1"] == r["truth"] or r["stage2_base"] == r["truth"])]
        rng = random.Random(20260907)
        ids = sorted(grouped)
        draws = []
        for _ in range(args.bootstrap):
            sample = [pair for _ in ids for pair in grouped[rng.choice(ids)]]
            value = auc(sample)
            if value is not None:
                draws.append(value)
        draws.sort()
        lo = draws[int(.025 * (len(draws) - 1))]
        hi = draws[int(.975 * (len(draws) - 1))]
        fix = [r for r in all_rows if r["stage2_truth_rank"] > 0]
        out["domains"][domain] = dict(input_sha256=done["blocks_sha256"],
            reachable_decisions=len(all_rows), band_n=len(band),
            band_repair=len(good), band_damage=len(bad), band_neither=len(band)-len(good)-len(bad),
            all_base_wrong_auc=auc([(r["stage2_p_wrong"], r["stage2_base"] != r["truth"]) for r in all_rows]),
            band_repair_vs_damage_auc=auc(pairs), band_auc_ci95=[lo, hi],
            bootstrap_valid=len(draws), bootstrap_undefined=args.bootstrap-len(draws),
            mean_detector_repair=statistics.mean(r["stage2_p_wrong"] for r in good),
            mean_detector_damage=statistics.mean(r["stage2_p_wrong"] for r in bad),
            reachable_fixable_n=len(fix),
            reachable_fixed=sum(r["stage2_pick"] == r["truth"] for r in fix),
            reachable_best_alternative_correct=sum(r["stage2_best_alt"] == r["truth"] for r in fix))
        print(domain, json.dumps(out["domains"][domain]), flush=True)
    with path.open("x") as stream:
        json.dump(out, stream, indent=2)
        stream.write("\n")
    print("DECISION_BOUNDARY_ANALYSIS_PASS", flush=True)


if __name__ == "__main__":
    main()
