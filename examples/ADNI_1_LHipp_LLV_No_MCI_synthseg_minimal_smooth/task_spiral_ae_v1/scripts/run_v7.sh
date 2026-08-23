#!/usr/bin/env bash
# v7: latent-128 head-to-head, capacity range shifted DOWN. One experiment per GPU; GPU 1 free.
#
#   GPU 0:  spiralnet_z128     GPU 2:  adaptive_z128
#
# Why down, not up. The v6 enqueued ladder (same geometry, only width varying) came out
# monotonically WORSE with width:
#       base   96 -> 0.037215 / 0.037502   (spiral / adaptive)
#             128 -> 0.037540 / 0.037922
#             160 -> 0.037706 / 0.039219
#             192 -> 0.038313 /    -
# So v5's clustering at base=112 was never a capacity ceiling; with 2037 training meshes wider
# models just overfit. v6's 64-256 range spent trials in a region already shown to be worse.
# v7 uses 32-128 step 16, which restores 112 (v5's exact winner, off the v6 grid) and extends
# BELOW 96 -- the direction the ladder points and the only part never tested.
#
# Budget raised 3600 -> 5400 s. At base 96 the adaptive op completed only 360/400 epochs, so
# the widening head-to-head gap at larger widths was confounded with truncation rather than
# being a property of the operator. 5400 s lets adaptive finish 400 epochs across this range,
# which removes the confound. Budget stays EQUAL for both architectures.
set -uo pipefail
TAG="${1:-v7}"
PY=/home/jakaria/anaconda3/envs/pytorch_geo/bin/python
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN=/mnt/bulk10tb/Deep3DComp/ADNI_1_LHipp/task_spiral_ae_v1/run
mkdir -p "$RUN"

# torch 2.0 has no expandable_segments; these two are supported and bound fragmentation.
export PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:0.8,max_split_size_mb:512

COMMON=(--n-trials 30 --trial-time-budget 5400 --min-epochs 150 --patience 10
        --seed-repeats 5 --base-min 32 --base-max 128 --base-step 16
        --epochs-min 300 --epochs-max 600 --dyn-frac-min 0.15 --dyn-frac-max 0.75)

start() {
  local EXP="$1" GPU="$2" PIDFILE="$RUN/v7_$1.pid"
  if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    echo "SKIP $EXP already running"; return; fi
  setsid nohup "$PY" -u "$HERE/optuna_search.py" --exp "$EXP" --tag "$TAG" --gpu "$GPU" \
      "${COMMON[@]}" >"$RUN/v7_${EXP}.out" 2>&1 </dev/null &
  echo $! >"$PIDFILE"; disown || true
  echo "START $EXP gpu=$GPU pid=$(cat "$PIDFILE")"
}

start spiralnet_z128 0
start adaptive_z128 2
echo; echo "gpu 1 untouched; v1/v2/v4/v5/v6 results untouched"
