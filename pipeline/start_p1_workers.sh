#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE="$HOME/abada-night"
GPUS="${1:-0,1,2,3,4,5,6,7}"
DAEMON_MODE="${DAEMON:-0}"

# Avoid CPU thread oversubscription across one worker per GPU.
export THREADS="${THREADS:-16}"
export OMP_NUM_THREADS="$THREADS"
export MKL_NUM_THREADS="$THREADS"
export OPENBLAS_NUM_THREADS="$THREADS"
export NUMEXPR_NUM_THREADS="$THREADS"
export TORCH_NUM_THREADS="$THREADS"
export TORCH_NUM_INTEROP_THREADS="${TORCH_NUM_INTEROP_THREADS:-1}"
export JOB_TIMEOUT_SEC="${JOB_TIMEOUT_SEC:-3600}"

cd "$SCRIPT_DIR"
mkdir -p "$BASE"/{backlog,running,done} logs

IFS=',' read -ra ARR <<< "$GPUS"
echo "P1 workers on GPUs: $GPUS (DAEMON=$DAEMON_MODE, THREADS=$THREADS, JOB_TIMEOUT_SEC=$JOB_TIMEOUT_SEC)"
for g in "${ARR[@]}"; do
  LOG="logs/p1_g${g}.log"
  CUDA_VISIBLE_DEVICES="$g" WORKER_ID="g$g" DAEMON="$DAEMON_MODE" WANDB_SILENT=true \
    setsid nohup python p1_worker.py >> "$LOG" 2>&1 &
  echo "  worker g$g -> pid $! -> $LOG"
done
echo "launched. tail $SCRIPT_DIR/logs/p1_g*.log"
