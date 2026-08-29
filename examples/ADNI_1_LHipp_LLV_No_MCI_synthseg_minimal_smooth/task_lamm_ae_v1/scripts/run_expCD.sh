#!/usr/bin/env bash
# Experiments C and D: close the FITTING gap, the only thing left between LAMM and SpiralNet++.
#
# expB settled generalisation: five reproduced trials at gap 1.38-1.46x against SpiralNet++'s
# 1.41x. The entire remaining deficit is fit -- train 0.029771 vs 0.026021, 14.4%. And the
# arithmetic is favourable: at LAMM's OWN gap of 1.38x, train 0.026021 gives 0.035909, which
# BEATS SpiralNet++'s 0.036784.
#
# THE DEFECT expC TARGETS. expB's best trials early-stopped at epoch 129-171 while the cosine
# was stretched over 450, so the lr was still at 70-83% of peak when training ended:
#     stopped at 129 -> lr 83.0% of peak   (30.7% on a 200-epoch schedule)
#     stopped at 171 -> lr 70.4% of peak   ( 5.6%)
# They never reached the low-lr annealing phase where fine detail is fit. `epochs` is searched
# so the schedule length matches where the model actually converges.
#
#   expC_fit        gpu 0  architecture FROZEN at expB's reproduced optimum (43,172 / D=192 /
#                          E6/D4 / bs16 / raw); searches schedule length and regularisation.
#   expD_finescale  gpu 1  less decoder compression still: 344 heads emit 54 coords from
#                          D=192 (0.28x) against 172's 0.5x and 43's 1.9x. TRANSFORMER ONLY --
#                          MLPMixer token-mixing is O(K^2) and hits 49.7M at 559 tokens where
#                          the transformer is 29.2M, so mixer would just be rejected.
set -uo pipefail
PY=/home/jakaria/anaconda3/envs/pytorch_geo/bin/python
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN=/mnt/bulk10tb/Deep3DComp/ADNI_1_LHipp/task_lamm_ae_v1/run
mkdir -p "$RUN"
[[ -d /mnt/bulk10tb/Deep3DComp ]] || { echo "FATAL: /mnt/bulk10tb is not mounted"; exit 1; }

# patience 60 with min-epochs 100 so a matched schedule RUNS TO COMPLETION rather than being
# early-stopped before it anneals -- the very failure this experiment exists to fix.
launch () {  # study space gpu backbone maxparams budget
  local S="$1" SP="$2" G="$3" BB="$4" MP="$5" TB="$6"
  if [[ -f "$RUN/$S.pid" ]] && kill -0 "$(cat "$RUN/$S.pid")" 2>/dev/null; then
    echo "SKIP $S (pid $(cat "$RUN/$S.pid"))"; return; fi
  setsid nohup "$PY" -u "$HERE/optuna_lamm.py" --study "$S" --space "$SP" --gpu "$G" \
      --backbone "$BB" --n-trials 16 --min-epochs 100 --patience 60 \
      --trial-time-budget "$TB" --max-params "$MP" --latent 128 \
      >>"$RUN/$S.out" 2>&1 </dev/null &
  sleep 3
  pgrep -f "optuna_lamm.py --study $S " | head -1 > "$RUN/$S.pid"
  echo "START $S space=$SP gpu=$G backbone=$BB pid=$(cat "$RUN/$S.pid")"
}

launch expC_fit       fit       0 search      26e6 2400
launch expD_finescale finescale 1 transformer 32e6 3400
echo; echo "logs: $RUN/exp[CD]_*.out"
