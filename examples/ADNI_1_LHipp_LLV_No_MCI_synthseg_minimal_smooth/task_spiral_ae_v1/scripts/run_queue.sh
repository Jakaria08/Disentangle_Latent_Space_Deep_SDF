#!/usr/bin/env bash
# v4 queue runner: exactly TWO GPU jobs at a time, on GPU 0 and GPU 2. GPU 1 is never used.
#
#   lane A (GPU 0):  spiralnet_z128 -> adaptive_z128 -> mlp_z128
#   lane B (GPU 2):  spiralnet_z256 -> adaptive_z256 -> mlp_z256
#
# Each lane is one detached `setsid` process running its experiments strictly sequentially:
# nothing starts on a GPU until the previous job on that GPU has exited. Each lane survives
# the terminal / SSH / Claude Code exiting.
#
# Restart-safe: experiments that wrote a .done marker are skipped, so re-running after a
# reboot resumes the queue and costs at most one experiment.
#
#   ./run_queue.sh            start (tag v4)
#   ./run_queue.sh v4b        start with a different tag
#   kill $(cat .../run/lane_A.pid)    stop a lane after its current experiment is killed

set -uo pipefail
TAG="${1:-v4}"
PY=/home/jakaria/anaconda3/envs/pytorch_geo/bin/python
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN=/mnt/bulk10tb/Deep3DComp/ADNI_1_LHipp/task_spiral_ae_v1/run
mkdir -p "$RUN"

N_TRIALS=40
SEED_REPEATS=5
BUDGET=1200

run_lane() {                       # $1 = lane id, $2 = gpu, $3.. = experiments in order
  local LANE="$1" GPU="$2"; shift 2
  for EXP in "$@"; do
    local MARK="$RUN/${EXP}_${TAG}.done"
    if [[ -f "$MARK" ]]; then
      echo "[$(date +%H:%M:%S)] lane $LANE: SKIP $EXP (already done)"
      continue
    fi
    echo "[$(date +%H:%M:%S)] lane $LANE: START $EXP on gpu $GPU"
    "$PY" -u "$HERE/search_v4.py" --exp "$EXP" --tag "$TAG" --gpu "$GPU" \
        --n-trials "$N_TRIALS" --seed-repeats "$SEED_REPEATS" \
        --trial-time-budget "$BUDGET" --patience 8
    local RC=$?
    if [[ $RC -eq 0 ]]; then
      touch "$MARK"
      echo "[$(date +%H:%M:%S)] lane $LANE: DONE $EXP"
    else
      echo "[$(date +%H:%M:%S)] lane $LANE: FAILED $EXP (rc=$RC) -- continuing to next"
    fi
  done
  echo "[$(date +%H:%M:%S)] lane $LANE: queue empty"
}

# re-entrant worker mode (invoked by the detached launcher below)
if [[ "${1:-}" == "--worker" ]]; then
  shift; TAG="$1"; LANE="$2"; GPU="$3"; shift 3
  run_lane "$LANE" "$GPU" "$@"
  exit 0
fi

start_lane() {                     # $1 = lane id, $2 = gpu, $3.. = experiments
  local LANE="$1" GPU="$2"; shift 2
  local PIDFILE="$RUN/lane_${LANE}.pid"
  if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    echo "SKIP lane $LANE already running (pid $(cat "$PIDFILE"))"; return
  fi
  setsid nohup bash "$HERE/run_queue.sh" --worker "$TAG" "$LANE" "$GPU" "$@" \
      >"$RUN/lane_${LANE}.out" 2>&1 </dev/null &
  echo $! >"$PIDFILE"
  disown || true
  echo "START lane $LANE  gpu=$GPU  pid=$(cat "$PIDFILE")  queue: $*"
}

start_lane A 0 spiralnet_z128 adaptive_z128 mlp_z128
start_lane B 2 spiralnet_z256 adaptive_z256 mlp_z256

echo
echo "two jobs max (gpu 0 + gpu 2); gpu 1 untouched"
echo "status: $HERE/status_v4.sh $TAG"
