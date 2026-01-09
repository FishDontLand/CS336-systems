#!/usr/bin/env bash
set -uo pipefail

LOG_DIR="/lambda/nfs/CS336/CS336-systems/outputs/memory_logs"
PYTHON="/lambda/nfs/CS336/CS336-systems/.venv/bin/python"

mkdir -p "$LOG_DIR"

# Fixed args as a string (exportable)
FIXED_ARGS="--vocab_size 10000 --d_model 2560 --d_ff 10240 --num_layer 32 --num_heads 32 --theta 10000 --batch_size 4 --warmup_steps 5 --num_measures 1 --record_memory_usage true"
export FIXED_ARGS

CONTEXT_LENGTH=(128 256 512 1024)
FORWARD_ONLY_CHOICES=(true false)
CAST_PRECISIONS=(true false)

for length in "${CONTEXT_LENGTH[@]}"; do
  SUB_DIR="$LOG_DIR/context_length_${length}"
  mkdir -p "$SUB_DIR"
  export SUB_DIR
  export LENGTH="$length"
  for forward_only in "${FORWARD_ONLY_CHOICES[@]}";do
    if [[ "$forward_only" == "true" ]]; then
      SUB_SUB_DIR="$SUB_DIR/forward_only"
    else
      SUB_SUB_DIR="$SUB_DIR/full_pass"
    fi
    mkdir -p "$SUB_SUB_DIR"
    export SUB_SUB_DIR
    ERROR_LOG="$SUB_SUB_DIR/errors.log"
    for cast_precision in "${CAST_PRECISIONS[@]}";do
      export USE_MIX_PRECISION="$cast_precision"
      if [[ "$forward_only" == "true" ]]; then
        echo "Run: --context_length $LENGTH --forward_only true --use_mixed_precision $USE_MIX_PRECISION"
        "$PYTHON" ../cs336_systems/monitoring.py $FIXED_ARGS --context_length $LENGTH --forward_only true --use_mixed_precision $USE_MIX_PRECISION --memory_log_path $SUB_SUB_DIR 2>>"$ERROR_LOG"
      else
        echo "Run: --context_length $LENGTH --use_mixed_precision $USE_MIX_PRECISION"
        "$PYTHON" ../cs336_systems/monitoring.py $FIXED_ARGS --context_length $LENGTH --use_mixed_precision $USE_MIX_PRECISION --memory_log_path $SUB_SUB_DIR 2>>"$ERROR_LOG"
      fi
      done
    done
done

echo "Done. Logs in: $LOG_DIR"