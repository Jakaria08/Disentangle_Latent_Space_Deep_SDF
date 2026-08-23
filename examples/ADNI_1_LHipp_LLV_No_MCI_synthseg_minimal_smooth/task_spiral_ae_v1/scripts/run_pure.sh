#!/usr/bin/env bash
# Pure SpiralNet++ / Adaptive Spiral vs PCA. Two GPU jobs at a time: GPU 0 and GPU 2.
# GPU 1 is never referenced.
#
#   lane A (GPU 0):  spiralnet_z128 -> adaptive_z128
#   lane B (GPU 2):  spiralnet_z256 -> adaptive_z256
#
# Fixes the three handicaps that crippled the v1 pure runs:
#   * 2400 s/trial (v1's 900 s truncated 3 of 4 best trials mid-training)
#   * epochs searched to 900 (was 600), min-epochs 150 before pruning/patience may fire
#   * 40 trials each (v1 had 12-18), regularisation axes searched, 5 seed repeats
#
#   ./run_pure.sh [TAG]      default TAG=v5
set -uo pipefail
TAG="${1:-v5}"
PY=/home/jakaria/anaconda3/envs/pytorch_geo/bin/python
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN=/mnt/bulk10tb/Deep3DComp/ADNI_1_LHipp/task_spiral_ae_v1/run
mkdir -p "$RUN"

N_TRIALS=40; BUDGET=2400; MIN_EPOCHS=150; SEEDS=5; PATIENCE=10

if [[ "${1:-}" == "--worker" ]]; then
  shift; TAG="$1"; LANE="$2"; GPU="$3"; shift 3
  for EXP in "$@"; do
    MARK="$RUN/${EXP}_${TAG}.done"
    if [[ -f "$MARK" ]]; then echo "[$(date +%H:%M:%S)] lane $LANE: SKIP $EXP"; continue; fi
    echo "[$(date +%H:%M:%S)] lane $LANE: START $EXP on gpu $GPU"
    "$PY" -u "$HERE/optuna_search.py" --exp "$EXP" --tag "$TAG" --gpu "$GPU" \
        --n-trials "$N_TRIALS" --trial-time-budget "$BUDGET" \
        --min-epochs "$MIN_EPOCHS" --patience "$PATIENCE" --seed-repeats "$SEEDS"
    RC=$?
    if [[ $RC -eq 0 ]]; then touch "$MARK"; echo "[$(date +%H:%M:%S)] lane $LANE: DONE $EXP"
    else echo "[$(date +%H:%M:%S)] lane $LANE: FAILED $EXP rc=$RC -- continuing"; fi
  done
  echo "[$(date +%H:%M:%S)] lane $LANE: queue empty"
  exit 0
fi

start_lane() {
  local LANE="$1" GPU="$2"; shift 2
  local PIDFILE="$RUN/pure_${LANE}.pid"
  if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    echo "SKIP lane $LANE already running"; return; fi
  setsid nohup bash "$HERE/run_pure.sh" --worker "$TAG" "$LANE" "$GPU" "$@" \
      >"$RUN/pure_${LANE}.out" 2>&1 </dev/null &
  echo $! >"$PIDFILE"; disown || true
  echo "START lane $LANE gpu=$GPU pid=$(cat "$PIDFILE") queue: $*"
}

start_lane A 0 spiralnet_z128 adaptive_z128
start_lane B 2 spiralnet_z256 adaptive_z256
echo; echo "two jobs max (gpu 0 + gpu 2); gpu 1 untouched"
