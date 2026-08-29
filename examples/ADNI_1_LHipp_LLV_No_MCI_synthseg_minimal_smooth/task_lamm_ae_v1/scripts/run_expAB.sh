#!/usr/bin/env bash
# Experiments A and B: shift LAMM's train/gap frontier, rather than slide along it.
#
# THE DIAGNOSIS. Sorting every LAMM trial by TRAINING error shows the deficit is not
# generalisation. At its best-val point LAMM's gap is 1.41x -- identical to SpiralNet++'s --
# and it has reached train 0.022724, which is 12.7% BETTER fitting than SpiralNet++'s
# 0.026021. What it cannot do is both at once: learning rate slides it monotonically along a
# fixed frontier across ten trials (lr 4.86e-4 -> train 0.0227/gap 2.00; lr 1.99e-4 ->
# train 0.0294/gap 1.41). Beating 0.036784 needs train <= 0.026088 AT gap 1.41 -- a point
# off that frontier, which no amount of lr tuning reaches.
#
#   expA_moments  gpu 0  region_mode raw vs both, K=86 fixed. The ONLY class of change that
#                        has ever shifted the frontier here is more encoder input: flatten
#                        gave 46% and raw-vs-moments gave 7.7% with training error unchanged
#                        to five decimals. LAMM tokenizes raw coordinates only.
#   expB_decoder  gpu 1  v_hat_i = W_i^out y_i^L makes 3*N_i coords from ONE D-vector:
#                        K=86 is 195/192 = 1.02x, exactly the compression limit -- and every
#                        well-fitting trial sits there. K=172 is 96/192 = 0.50x, free. Every
#                        set here either raises D or adds a fine scale that cannot compress.
set -uo pipefail
PY=/home/jakaria/anaconda3/envs/pytorch_geo/bin/python
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN=/mnt/bulk10tb/Deep3DComp/ADNI_1_LHipp/task_lamm_ae_v1/run
mkdir -p "$RUN"
[[ -d /mnt/bulk10tb/Deep3DComp ]] || { echo "FATAL: /mnt/bulk10tb is not mounted"; exit 1; }

COMMON=(--backbone search --n-trials 14 --epochs 450 --min-epochs 80 --patience 30
        --trial-time-budget 3000 --max-params 26e6 --latent 128)

launch () {  # study space gpu
  local S="$1" SP="$2" G="$3"
  if [[ -f "$RUN/$S.pid" ]] && kill -0 "$(cat "$RUN/$S.pid")" 2>/dev/null; then
    echo "SKIP $S (pid $(cat "$RUN/$S.pid"))"; return; fi
  setsid nohup "$PY" -u "$HERE/optuna_lamm.py" --study "$S" --space "$SP" --gpu "$G" \
      "${COMMON[@]}" >>"$RUN/$S.out" 2>&1 </dev/null &
  sleep 3
  pgrep -f "optuna_lamm.py --study $S " | head -1 > "$RUN/$S.pid"
  echo "START $S space=$SP gpu=$G pid=$(cat "$RUN/$S.pid")"
}

launch expA_moments moments 0
launch expB_decoder decoder 1
echo; echo "logs: $RUN/exp[AB]_*.out"
