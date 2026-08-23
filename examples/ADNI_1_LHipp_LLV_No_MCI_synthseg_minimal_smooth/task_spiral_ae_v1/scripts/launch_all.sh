#!/usr/bin/env bash
# Launch the four Optuna studies fully detached.
#
# Each study runs under `setsid` in its own session, so it survives the terminal, the SSH
# connection, and the Claude Code process exiting -- the v1 runs died because they were still
# children of the agent shell's process group.
#
# Usage:  ./launch_all.sh [TAG]      (default TAG=v2)
# Stop:   kill $(cat /mnt/bulk10tb/Deep3DComp/ADNI_1_LHipp/task_spiral_ae_v1/run/*.pid)
# Resume: re-run this script with the same TAG; Optuna continues each SQLite study.

set -euo pipefail

TAG="${1:-v2}"
PY=/home/jakaria/anaconda3/envs/pytorch_geo/bin/python
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR=/mnt/bulk10tb/Deep3DComp/ADNI_1_LHipp/task_spiral_ae_v1/run
mkdir -p "$RUN_DIR"

# exp:gpu:trials:trial_time_budget_s
JOBS=(
  "spiralnet_z128:0:80:1200"
  "spiralnet_z256:0:80:1200"
  "adaptive_z128:1:60:1500"
  "adaptive_z256:2:60:1500"
)

for job in "${JOBS[@]}"; do
  IFS=: read -r EXP GPU TRIALS BUDGET <<<"$job"
  KEY="${EXP}_${TAG}"
  PIDFILE="$RUN_DIR/${KEY}.pid"

  if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    echo "SKIP  $KEY already running (pid $(cat "$PIDFILE"))"
    continue
  fi

  setsid nohup "$PY" -u "$HERE/optuna_search.py" \
      --exp "$EXP" --tag "$TAG" --gpu "$GPU" \
      --n-trials "$TRIALS" --trial-time-budget "$BUDGET" --patience 8 \
      >"$RUN_DIR/${KEY}.out" 2>&1 </dev/null &
  echo $! >"$PIDFILE"
  disown || true
  echo "START $KEY  gpu=$GPU trials=$TRIALS budget=${BUDGET}s pid=$(cat "$PIDFILE")"
done

echo
echo "detached. logs: $RUN_DIR/*.out  and  .../logs/<exp>_${TAG}/*.log"
echo "status:  $HERE/status.sh $TAG"
