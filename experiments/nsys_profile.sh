#!/usr/bin/env bash
set -uo pipefail

PARAM_FILE="params.txt"
LOG_DIR="/lambda/nfs/CS336/CS336-systems/outputs/profile_logs"
MAX_PARALLEL=1

mkdir -p "$LOG_DIR"

# Fixed args as a string (exportable)
FIXED_ARGS="--vocab_size 10000 --theta 10000 --batch_size 4 --warmup_steps 0 --num_measures 10"
export FIXED_ARGS

CONTEXT_LENGTH=(128 256 512 1024)


for length in "${CONTEXT_LENGTH[@]}"; do
  SUB_DIR="$LOG_DIR/context_length_${length}"
  mkdir -p "$SUB_DIR"
  export SUB_DIR
  export LENGTH="$length"
  nl -ba "$PARAM_FILE" | xargs -P "$MAX_PARALLEL" -I{} bash -lc '
  set -uo pipefail
  line_no=$(echo "{}" | awk "{print \$1}")
  args=$(echo "{}" | cut -f2-)

  echo "Run ${line_no}: ${args} --context_length ${LENGTH}"
  nsys profile --force-overwrite=true -o "$SUB_DIR/result_${LENGTH}_${line_no}" python ../cs336_systems/monitoring.py $FIXED_ARGS --context_length $LENGTH ${args}
'
done

echo "Done. Logs in: $LOG_DIR"