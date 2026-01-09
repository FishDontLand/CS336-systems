#!/usr/bin/env bash
set -euo pipefail

PARAM_FILE="params.txt"
LOG_DIR="/lambda/nfs/CS336/CS336-systems/outputs/benchmark_logs"
MAX_PARALLEL=1
PYTHON="/lambda/nfs/CS336/CS336-systems/.venv/bin/python"
export PYTHON

mkdir -p "$LOG_DIR"

# Fixed args as a string (exportable)
FIXED_ARGS="--vocab_size 10000 --theta 10000 --context_length 256 --batch_size 4 --warmup_steps 5 --num_measures 10 --use_mixed_precision true"
export FIXED_ARGS

nl -ba "$PARAM_FILE" | xargs -P "$MAX_PARALLEL" -I{} bash -lc '
  set -euo pipefail
  line_no=$(echo "{}" | awk "{print \$1}")
  args=$(echo "{}" | cut -f2-)

  out="'"$LOG_DIR"'/run_${line_no}_mixed_precision.out"

  echo "Run ${line_no}: ${args}"
  "$PYTHON" ../cs336_systems/monitoring.py $FIXED_ARGS ${args} >"${out}"
'
echo "Done. Logs in: $LOG_DIR"
