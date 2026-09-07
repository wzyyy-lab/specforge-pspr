#!/usr/bin/env bash
# Compact training monitor: one line per logged step, only the fields that can reveal a broken run.
LOG=${1:-outputs/PSPR_V2_STAGE1.log}
N=${2:-8}
tr '\r' '\n' < "$LOG" | grep -a "^step " | awk '!seen[$2]++' | tail -"$N" | python3 -c "
import sys, ast
print(f\"{'step':>6} {'sel_loss':>9} {'sel_acc':>8} {'cover':>7} {'err_acc':>8} {'alpha':>6} {'gnorm':>8} {'lr':>9} {'step/h':>8}\")
for line in sys.stdin:
    line = line.strip()
    if not line: continue
    step = line.split(':')[0].split()[1]
    d = ast.literal_eval(line.split(': ', 1)[1])
    print(f\"{step:>6} {d['train/selector_loss']:9.4f} {d['train/selector_accuracy']:8.4f} \"
          f\"{d['train/selector_coverage']:7.4f} {d['train/selector_err_accuracy']:8.4f} \"
          f\"{d['train/selector_loss_alpha']:6.2f} {d['train/grad_norm']:8.4f} \"
          f\"{d['train/lr']:9.2e} {d['train/perf/optimizer_steps_per_hour']:8.0f}\")
"
