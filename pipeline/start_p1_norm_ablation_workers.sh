#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE="${ABADA_BASE:-$HOME/abada-night/p1_norm_ablation}"
HOST="$(hostname)"
DAEMON_MODE="${DAEMON:-0}"

if [[ "$HOST" == *n1* || "$HOST" == *wku-vs-01* ]]; then
  DEFAULT_GPUS="4,5,6,7"
else
  DEFAULT_GPUS="0,1,2,3,4,5,6,7"
fi
GPUS="${1:-$DEFAULT_GPUS}"

IFS=',' read -ra GPU_ARR <<< "$GPUS"
if [[ "$HOST" == *n1* || "$HOST" == *wku-vs-01* ]]; then
  for g in "${GPU_ARR[@]}"; do
    case "$g" in
      0|1|2|3)
        echo "ERROR: n1 GPU 0-3 are reserved. Use GPUs 4,5,6,7 only." >&2
        exit 2
        ;;
    esac
  done
fi

export ABADA_BASE="$BASE"
export THREADS="${THREADS:-16}"
export OMP_NUM_THREADS="$THREADS"
export MKL_NUM_THREADS="$THREADS"
export OPENBLAS_NUM_THREADS="$THREADS"
export NUMEXPR_NUM_THREADS="$THREADS"
export TORCH_NUM_THREADS="$THREADS"
export TORCH_NUM_INTEROP_THREADS="${TORCH_NUM_INTEROP_THREADS:-1}"
export JOB_TIMEOUT_SEC="${JOB_TIMEOUT_SEC:-3600}"
export WANDB_SILENT="${WANDB_SILENT:-true}"

cd "$SCRIPT_DIR"
mkdir -p "$BASE"/{backlog,running,done} logs

echo "P1 normalization-ablation workers"
echo "  host=$HOST"
echo "  base=$BASE"
echo "  gpus=$GPUS"
echo "  daemon=$DAEMON_MODE"
echo "  threads=$THREADS"
echo "  job_timeout=$JOB_TIMEOUT_SEC"

for g in "${GPU_ARR[@]}"; do
  LOG="logs/p1_norm_g${g}.log"
  CUDA_VISIBLE_DEVICES="$g" WORKER_ID="norm_g$g" DAEMON="$DAEMON_MODE" \
    setsid nohup python p1_worker.py >> "$LOG" 2>&1 &
  echo "  worker g$g -> pid $! -> $LOG"
done

echo "launched. tail $SCRIPT_DIR/logs/p1_norm_g*.log"
