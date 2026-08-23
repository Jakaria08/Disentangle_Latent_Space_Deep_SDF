#!/usr/bin/env bash
# v6: latent-128 head-to-head at raised capacity. One experiment per GPU, GPU 1 untouched.
#
#   GPU 0:  spiralnet_z128     GPU 2:  adaptive_z128
#
# Why the space changed from v5: the v5 top-8 trials saturated at base_channels=112 of a 128
# ceiling, and both winners used ds=[2,4,4] (the widest pre-latent funnel). So base_channels
# is raised to 64-256. Measured peaks at base=256, bs=16: 3.4 GB spiral, 12.9 GB adaptive --
# both fit one-per-GPU.
#
# The adaptive op is ~17x slower per iteration at base=256 (203 vs 12 ms). The budget is kept
# EQUAL for both rather than compensated: that cost is a real property of the operator, and
# epochs_completed + duration_s are recorded per trial so it shows up in the report.
#
# Identical search space for both architectures -- this is a head-to-head.
set -uo pipefail
TAG="${1:-v6}"
PY=/home/jakaria/anaconda3/envs/pytorch_geo/bin/python
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN=/mnt/bulk10tb/Deep3DComp/ADNI_1_LHipp/task_spiral_ae_v1/run
mkdir -p "$RUN"

COMMON=(--n-trials 30 --trial-time-budget 3600 --min-epochs 150 --patience 10
        --seed-repeats 5 --base-min 64 --base-max 256 --base-step 32
        --epochs-min 300 --epochs-max 600 --dyn-frac-min 0.15 --dyn-frac-max 0.75)

start() {                      # $1 exp, $2 gpu
  local EXP="$1" GPU="$2" PIDFILE="$RUN/v6_$1.pid"
  if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    echo "SKIP $EXP already running"; return; fi
  setsid nohup "$PY" -u "$HERE/optuna_search.py" --exp "$EXP" --tag "$TAG" --gpu "$GPU" \
      "${COMMON[@]}" >"$RUN/v6_${EXP}.out" 2>&1 </dev/null &
  echo $! >"$PIDFILE"; disown || true
  echo "START $EXP gpu=$GPU pid=$(cat "$PIDFILE")"
}

start spiralnet_z128 0
start adaptive_z128 2
echo; echo "gpu 1 untouched; v1/v2/v4/v5 results untouched"
