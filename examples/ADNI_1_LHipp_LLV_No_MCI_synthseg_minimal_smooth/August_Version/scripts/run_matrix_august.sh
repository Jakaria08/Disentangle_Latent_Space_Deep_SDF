#!/usr/bin/env bash
# August_Version 3x3 matrix: {pca128, spiralnet128, adaptive128} x {direct_c4, plain_ode, brainode}
#
# Four lanes, two per GPU, GPU 1 left free for the user's other work. Each GPU carries one
# cocycle (direct_c4) lane and one ODE lane, as requested, so the two job types share a card
# rather than two heavy jobs of the same kind contending.
#
#   GPU 0 lane C0 (cocycle): pca128, adaptive128        direct_c4
#   GPU 0 lane O0 (ode):     pca128, spiralnet128, adaptive128   plain_ode
#   GPU 2 lane C2 (cocycle): spiralnet128               direct_c4
#   GPU 2 lane O2 (ode):     pca128, spiralnet128, adaptive128   brainode
#
# Lanes run their jobs sequentially; .done markers make the whole thing restart-safe.
set -uo pipefail
TAG="${1:-s42}"
SEED="${2:-42}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK="examples/ADNI_1_LHipp_LLV_No_MCI_synthseg_minimal_smooth/August_Version"
OUT=/mnt/bulk10tb/Deep3DComp/ADNI_1_LHipp/August_Version
RUN=$OUT/run; mkdir -p "$RUN"

if [[ "${1:-}" == "--worker" ]]; then
  shift; LANE="$1"; GPU="$2"; SEED="$3"; TAG="$4"; shift 4
  for JOB in "$@"; do                      # JOB = rep:method
    REP="${JOB%%:*}"; METHOD="${JOB##*:}"
    NAME="${REP}_${METHOD}_${TAG}"
    MARK="$RUN/${NAME}.done"
    [[ -f "$MARK" ]] && { echo "[$(date +%H:%M:%S)] $LANE SKIP $NAME"; continue; }
    echo "[$(date +%H:%M:%S)] $LANE START $NAME on gpu $GPU"
    "$HERE/run_one_gpu.sh" "$TASK/configs/august_${REP}_${METHOD}.json" "$GPU" "$SEED" "$NAME"
    if [[ $? -eq 0 ]]; then touch "$MARK"; echo "[$(date +%H:%M:%S)] $LANE DONE $NAME"
    else echo "[$(date +%H:%M:%S)] $LANE FAILED $NAME -- continuing"; fi
  done
  echo "[$(date +%H:%M:%S)] $LANE queue empty"; exit 0
fi

start() {                                   # $1 lane, $2 gpu, rest = jobs
  local LANE="$1" GPU="$2"; shift 2
  local PID="$RUN/lane_${LANE}_${TAG}.pid"
  if [[ -f "$PID" ]] && kill -0 "$(cat "$PID")" 2>/dev/null; then
    echo "SKIP lane $LANE already running"; return; fi
  setsid nohup bash "$HERE/run_matrix_august.sh" --worker "$LANE" "$GPU" "$SEED" "$TAG" "$@" \
      >"$RUN/lane_${LANE}_${TAG}.out" 2>&1 </dev/null &
  echo $! >"$PID"; disown || true
  echo "START lane $LANE gpu=$GPU pid=$(cat "$PID") jobs: $*"
}

start C0 0 pca128:direct_c4 adaptive128:direct_c4
start O0 0 pca128:plain_ode spiralnet128:plain_ode adaptive128:plain_ode
start C2 2 spiralnet128:direct_c4
start O2 2 pca128:brainode spiralnet128:brainode adaptive128:brainode
echo; echo "4 lanes, 2 per GPU on 0 and 2; GPU 1 untouched"
