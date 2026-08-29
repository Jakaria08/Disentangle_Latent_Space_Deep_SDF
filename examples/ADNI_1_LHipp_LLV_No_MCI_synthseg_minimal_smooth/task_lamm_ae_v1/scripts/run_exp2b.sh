#!/usr/bin/env bash
# exp2b: the THIRD term of the residual decomposition, X = X_coarse + R_medium + R_fine.
#
# Replaces the original exp2 (unboxed capacity at K=86), which was stopped after 9 trials at
# best 0.057052 against v1's 0.041489. Post-mortem: v1's winning config WAS reachable there,
# and dim_head cannot explain the gap because MLPMixer has no attention at all -- but 8 of 10
# trials drew batch_size=32, and v1's data had bs=16 winning 11 of 16 top trials. That study
# was measuring a batch-size accident, not capacity.
#
# What exp1 established instead, and what this builds on:
#   * multiscale wins: three of four `11,43` trials beat EVERY single-scale trial, and it
#     holds at matched capacity (~12-13M: `11,43` 0.0759 vs single 43 0.1215, 172 0.1261)
#   * TPE re-derived the same 17.48M config twice (0.046525, 0.048044)
#   * K=86 -- the granularity every earlier run used -- is the WORST single scale tested
#   * exp1 never completed a THREE-scale set: all were rejected by the 20M guard
#     (30.99M, 38.31M, 20.54M). Three tokenizers + three head sets cost ~18M at D=192 before
#     the backbone, so 20M rejected the entire space. Hence --max-params 26M here.
set -uo pipefail
PY=/home/jakaria/anaconda3/envs/pytorch_geo/bin/python
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN=/mnt/bulk10tb/Deep3DComp/ADNI_1_LHipp/task_lamm_ae_v1/run
mkdir -p "$RUN"
[[ -d /mnt/bulk10tb/Deep3DComp ]] || { echo "FATAL: /mnt/bulk10tb is not mounted"; exit 1; }

S=exp2b_multiscale3
if [[ -f "$RUN/$S.pid" ]] && kill -0 "$(cat "$RUN/$S.pid")" 2>/dev/null; then
  echo "SKIP $S (pid $(cat "$RUN/$S.pid"))"; exit 0; fi
# 450 epochs / 3000s: measured at 1-4 s/epoch across the space, so 450 epochs costs
# 450-1800s and nothing is truncated. exp1's best two trials stopped at 239 and 546.
setsid nohup "$PY" -u "$HERE/optuna_lamm.py" --study "$S" --space scales3 \
    --backbone search --gpu 2 --n-trials 14 --epochs 450 --min-epochs 80 --patience 30 \
    --trial-time-budget 3000 --max-params 26e6 --latent 128 \
    >>"$RUN/$S.out" 2>&1 </dev/null &
sleep 3
pgrep -f "optuna_lamm.py --study $S " | head -1 > "$RUN/$S.pid"
echo "START $S space=scales3 gpu=2 pid=$(cat "$RUN/$S.pid")"
