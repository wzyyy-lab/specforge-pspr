#!/usr/bin/env bash
# Fine margin sweep for PSPR-Decision, requested after the coarse {0,0.5,1,2} scan on gsm8k n=8
# showed margin=0 winning there.  Two reasons that scan is not enough to settle the question:
#   * n=8 on one domain cannot resolve the ~0.05 differences this knob produces;
#   * the coarse grid jumps straight from 0 to 0.5, i.e. from 48.4% repair / 47 destroy to
#     29.1% / 1.  Everything interesting -- the point where marginal recovered stops paying for
#     marginal destroyed -- is strictly inside that gap.
#
# Every run here carries --allow-policy-mismatch (any margin != 0 is not the exported native
# policy), so these are CALIBRATION numbers, not reportable results.  They are measured on the
# same reserved prompts used for reporting, which is exactly the set-reuse problem noted in the
# log; the sweep is kept for the mechanism it reveals, not as a headline number.
set -u
SF=/home/wangzhuoyu/kl_infra_infer_intern/wangzhuoyu/SpecForge
STEP=${STEP:-5600}
MARGINS=${MARGINS:-"0 0.1 0.2 0.3 0.4 0.5"}
SAMPLES=${SAMPLES:-20}

for M in $MARGINS; do
  TAG=$(echo "$M" | tr -d '.')
  echo "### margin=$M"
  SUF="_m${TAG}" SAMPLES="$SAMPLES" bash "$SF/scripts/par_eval_decision.sh" "$STEP" "$M"
done

echo
echo "=== macro summary (step$STEP) ==="
for M in $MARGINS; do
  TAG=$(echo "$M" | tr -d '.')
  python3 - "$SF" "$STEP" "_m${TAG}" "$M" <<'PYEOF'
import pathlib
import re
import sys

sf, step, suf, margin = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
doms = ["gsm8k", "math500", "humaneval", "mbpp", "alpaca", "mt-bench"]
vals, extra = [], {}
for d in doms:
    p = pathlib.Path(sf) / "outputs" / f"EVAL_decision_step{step}{suf}_{d}.log"
    if not p.exists():
        vals.append(None)
        continue
    txt = p.read_text(errors="replace").replace("\r", "\n")
    hit = re.findall(rf"\[{re.escape(d)}\][^\n]*latrepair[^\n]*?([0-9]+\.[0-9]+)", txt)
    vals.append(float(hit[-1]) if hit else None)
    for key in ("base_right_kept", "fix_recovered", "destroy"):
        m = re.findall(rf"{key}[^0-9\n]*([0-9.]+)", txt)
        if m:
            extra.setdefault(key, []).append(m[-1])
ok = [v for v in vals if v is not None]
macro = sum(ok) / len(ok) if ok else float("nan")
cells = " ".join("  n/a" if v is None else f"{v:6.3f}" for v in vals)
print(f"margin={margin:>4}  macro={macro:7.4f}  [{cells}]  ({len(ok)}/6 domains)")
PYEOF
done
