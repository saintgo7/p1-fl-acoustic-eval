#!/usr/bin/env bash
set -euo pipefail

INTERVAL_SEC="${INTERVAL_SEC:-600}"
HOSTS="${HOSTS:-master n3}"

status_host() {
  local host="$1"
  ssh "$host" bash -s <<'REMOTE'
set -euo pipefail
BASE="$HOME/abada-night/p1_norm_ablation"
P1="$HOME/abada-night/p1"

count_done() {
  find "$BASE/done" -maxdepth 1 -type f -name '*.json' ! -name 'FAILED_*' ! -name 'DUP_*' | wc -l | tr -d ' '
}

worker_count() {
  ps -eo args | awk '$1=="python" && $2=="p1_worker.py" {n++} END{print n+0}'
}

watchdog_count() {
  ps -eo args | awk '$1=="bash" && $2=="./p1_norm_watchdog.sh" {n++} END{print n+0}'
}

printf 'host=%s\n' "$(hostname)"
date -Is
printf 'backlog=%s running=%s done=%s failed=%s workers=%s watchdogs=%s\n' \
  "$(find "$BASE/backlog" -maxdepth 1 -type f -name 'p1_norm_*.json' | wc -l | tr -d ' ')" \
  "$(find "$BASE/running" -maxdepth 1 -type f -name '*.json' | wc -l | tr -d ' ')" \
  "$(count_done)" \
  "$(find "$BASE/done" -maxdepth 1 -type f -name 'FAILED_*.json' | wc -l | tr -d ' ')" \
  "$(worker_count)" \
  "$(watchdog_count)"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader | sed 's/^/gpu=/'
fi

printf 'watchdog_tail\n'
tail -n 3 "$P1/logs/p1_norm_watchdog.log" 2>/dev/null || true
REMOTE
}

status_once() {
  printf '===== P1 norm status local=%s interval=%ss =====\n' "$(date '+%Y-%m-%dT%H:%M:%S%z')" "$INTERVAL_SEC"
  for host in $HOSTS; do
    status_host "$host"
    printf '%s\n' '---'
  done
}

status_once
while true; do
  sleep "$INTERVAL_SEC"
  status_once
done
