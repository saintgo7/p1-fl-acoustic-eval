#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE="${ABADA_BASE:-$HOME/abada-night/p1_norm_ablation}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
INTERVAL_SEC="${INTERVAL_SEC:-600}"
STALE_SEC="${STALE_SEC:-7200}"
JOB_TIMEOUT_SEC="${JOB_TIMEOUT_SEC:-5400}"
DAEMON="${DAEMON:-1}"
HOST="$(hostname)"

BACKLOG="$BASE/backlog"
RUNNING="$BASE/running"
DONE="$BASE/done"
LOG_DIR="$SCRIPT_DIR/logs"

mkdir -p "$BACKLOG" "$RUNNING" "$DONE" "$LOG_DIR"

IFS=',' read -ra GPU_ARR <<< "$GPUS"
EXPECTED_WORKERS="${EXPECTED_WORKERS:-${#GPU_ARR[@]}}"

if [[ "$HOST" == *n1* || "$HOST" == *wku-vs-01* ]]; then
  for g in "${GPU_ARR[@]}"; do
    case "$g" in
      0|1|2|3)
        echo "$(date -Is) ERROR n1 GPU 0-3 are reserved; refusing to start watchdog with GPUS=$GPUS" >&2
        exit 2
        ;;
    esac
  done
fi

job_name_from_claim() {
  local base="$1"
  base="${base#FAILED_}"
  base="$(echo "$base" | sed -E 's/^norm_g[0-9]+_//; s/^g[0-9]+_//')"
  echo "$base"
}

count_jobs() {
  local dir="$1"
  find "$dir" -maxdepth 1 -type f -name 'p1_norm_*.json' | wc -l | tr -d ' '
}

count_failed() {
  find "$DONE" -maxdepth 1 -type f -name 'FAILED_*.json' | wc -l | tr -d ' '
}

worker_count() {
  ps -eo args | awk '$1=="python" && $2=="p1_worker.py" {n++} END{print n+0}'
}

requeue_one() {
  local path="$1"
  local name target
  name="$(job_name_from_claim "$(basename "$path")")"
  target="$BACKLOG/$name"
  if [[ ! "$name" == p1_norm_*.json ]]; then
    echo "$(date -Is) skip non-job file: $path"
    return 0
  fi
  if [[ -e "$target" ]]; then
    echo "$(date -Is) target already exists, archiving duplicate: $path -> $DONE/DUP_$(basename "$path")"
    mv "$path" "$DONE/DUP_$(basename "$path")"
  else
    echo "$(date -Is) requeue: $path -> $target"
    mv "$path" "$target"
  fi
}

requeue_failed() {
  local found=0
  while IFS= read -r -d '' path; do
    found=1
    requeue_one "$path"
  done < <(find "$DONE" -maxdepth 1 -type f -name 'FAILED_*.json' -print0)
  return "$found"
}

requeue_stale_running() {
  local workers
  workers="$(worker_count)"
  if [[ "$workers" -gt 0 ]]; then
    echo "$(date -Is) skip stale-running requeue: workers_active=$workers"
    return 0
  fi
  local stale_min=$((STALE_SEC / 60))
  while IFS= read -r -d '' path; do
    requeue_one "$path"
  done < <(find "$RUNNING" -maxdepth 1 -type f -name '*.json' -mmin +"$stale_min" -print0)
}

requeue_all_running() {
  while IFS= read -r -d '' path; do
    requeue_one "$path"
  done < <(find "$RUNNING" -maxdepth 1 -type f -name '*.json' -print0)
}

restart_workers() {
  echo "$(date -Is) restarting workers: expected=$EXPECTED_WORKERS actual=$(worker_count)"
  pkill -f '[p]ython p1_worker.py' || true
  sleep 5
  requeue_all_running
  (
    cd "$SCRIPT_DIR"
    ABADA_BASE="$BASE" DAEMON="$DAEMON" JOB_TIMEOUT_SEC="$JOB_TIMEOUT_SEC" \
      ./start_p1_norm_ablation_workers.sh "$GPUS"
  )
}

echo "$(date -Is) watchdog start host=$HOST base=$BASE gpus=$GPUS interval=${INTERVAL_SEC}s stale=${STALE_SEC}s expected_workers=$EXPECTED_WORKERS"

while true; do
  requeue_failed || true
  requeue_stale_running || true

  backlog="$(count_jobs "$BACKLOG")"
  running="$(find "$RUNNING" -maxdepth 1 -type f -name '*.json' | wc -l | tr -d ' ')"
  done_count="$(find "$DONE" -maxdepth 1 -type f -name '*.json' ! -name 'FAILED_*' ! -name 'DUP_*' | wc -l | tr -d ' ')"
  failed="$(count_failed)"
  workers="$(worker_count)"

  echo "$(date -Is) status backlog=$backlog running=$running done=$done_count failed=$failed workers=$workers"

  if [[ "$backlog" -eq 0 && "$running" -eq 0 && "$failed" -eq 0 ]]; then
    echo "$(date -Is) COMPLETE all jobs finished; stopping remaining workers"
    pkill -f '[p]ython p1_worker.py' || true
    exit 0
  fi

  if [[ "$workers" -lt "$EXPECTED_WORKERS" ]]; then
    restart_workers
  fi

  sleep "$INTERVAL_SEC"
done
