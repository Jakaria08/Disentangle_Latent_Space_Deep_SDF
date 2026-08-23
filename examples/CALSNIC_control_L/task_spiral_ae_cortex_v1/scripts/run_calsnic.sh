#!/usr/bin/env bash
# CALSNIC left pial surface: SpiralNet++ / Adaptive Spiral vs PCA.
# Two GPU jobs at a time, GPU 0 and GPU 2. GPU 1 is never referenced.
#
#   lane A (GPU 0):  calsnic_spiralnet_z128 -> calsnic_adaptive_z128
#   lane B (GPU 2):  calsnic_spiralnet_z256 -> calsnic_adaptive_z256
#
# Run ONE job per GPU: the widest configurations peak near 11.5 GB at 40,962 vertices.
# Prerequisite (already done): fit_pca_calsnic.py, which gates on reproducing the published
# PCA numbers and writes the basis used as the paired comparator.
#
#   ./run_calsnic.sh [TAG]     default TAG=v1
set -uo pipefail
TAG="${1:-v1}"
PY=/home/jakaria/anaconda3/envs/pytorch_geo/bin/python
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN=/mnt/bulk10tb/Deep3DComp/CALSNIC/spiral_ae_cortex_v1/run
mkdir -p "$RUN"

N_TRIALS=40; BUDGET=2400; MIN_EPOCHS=150; SEEDS=5; PATIENCE=10

if [[ "${1:-}" == "--worker" ]]; then
  shift; TAG="$1"; LANE="$2"; GPU="$3"; shift 3
  for EXP in "$@"; do
    MARK="$RUN/${EXP}_${TAG}.done"
    [[ -f "$MARK" ]] && { echo "[$(date +%H:%M:%S)] lane $LANE: SKIP $EXP"; continue; }
    echo "[$(date +%H:%M:%S)] lane $LANE: START $EXP on gpu $GPU"
    "$PY" -u "$HERE/search_calsnic.py" --exp "$EXP" --tag "$TAG" --gpu "$GPU" \
        --n-trials "$N_TRIALS" --trial-time-budget "$BUDGET" \
        --min-epochs "$MIN_EPOCHS" --patience "$PATIENCE" --seed-repeats "$SEEDS"
    if [[ $? -eq 0 ]]; then touch "$MARK"; echo "[$(date +%H:%M:%S)] lane $LANE: DONE $EXP"
    else echo "[$(date +%H:%M:%S)] lane $LANE: FAILED $EXP -- continuing"; fi
  done
  echo "[$(date +%H:%M:%S)] lane $LANE: queue empty"; exit 0
fi

start_lane() {
  local LANE="$1" GPU="$2"; shift 2
  local PIDFILE="$RUN/lane_${LANE}.pid"
  if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    echo "SKIP lane $LANE already running"; return; fi
  setsid nohup bash "$HERE/run_calsnic.sh" --worker "$TAG" "$LANE" "$GPU" "$@" \
      >"$RUN/lane_${LANE}.out" 2>&1 </dev/null &
  echo $! >"$PIDFILE"; disown || true
  echo "START lane $LANE gpu=$GPU pid=$(cat "$PIDFILE") queue: $*"
}

start_lane A 0 calsnic_spiralnet_z128 calsnic_adaptive_z128
start_lane B 2 calsnic_spiralnet_z256 calsnic_adaptive_z256
echo; echo "two jobs max (gpu 0 + gpu 2); gpu 1 untouched"
