#!/bin/bash
# Pull the metrics that decide whether the cloze run is healthy out of the training log.
LOG=${1:-outputs/PSPR_CLOZE.log}
N=${2:-6}
tr '\r' '\n' < "$LOG" | grep -oE "step [0-9]+: \{.*" | tail -n "$N" | python3 -c "
import sys, ast
for line in sys.stdin:
    step = line.split(':',1)[0].split()[1]
    d = ast.literal_eval(line.split(': ',1)[1])
    g = lambda k, f='.4f': format(d[k], f) if k in d else '--'
    print(f\"step {step:>5}  sel_acc {g('train/selector_accuracy')}  cover {g('train/selector_coverage')}  \"
          f\"sel_loss {g('train/selector_loss')}  err_acc {g('train/selector_err_accuracy')}  \"
          f\"err_loss {g('train/selector_err_loss')}  gnorm {g('train/grad_norm')}  \"
          f\"lr {g('train/lr','.2e')}  s/step {g('train/perf/optimizer_step_time_s','.2f')}\")
"
