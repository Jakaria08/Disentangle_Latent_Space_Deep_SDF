#!/usr/bin/env bash
# Experiments 1 and 2, from the analysis of the v1 LAMM searches.
#
# WHAT v1 ACTUALLY MEASURED. Its winners sat ON the boundary in three dimensions
# (dec_depth=4 = my max, enc_depth=6 = my max, batch=16 = my min); the mixer's median best
# epoch was 377 of 400; half the trials hit the 1200s wall; D=256 got 1-2 trials; and 10 of
# 40 trials were spent on share_regions=True / id_token, which scored 0.16-0.22 every time.
# LAMM's own setting is D=512 for 1500 epochs. So 0.041489 describes the box, not the model.
#
# TWO IMPLEMENTATION DEVIATIONS FOUND, both fixed here:
#   * dim_head was dim//heads, giving inner width ~= D (192 at the winning D=192) where LAMM
#     fixes dim_head=64 -> inner 512. 2.7x narrower attention, and it also made the head
#     count meaningless since both 4 and 8 gave the same inner width.
#   * regions were 86 on 2746 vertices; LAMM splits 12k vertices into ELEVEN. An 8x longer
#     token sequence than the paper, never examined.
#
#   exp1_scales  gpu 0  region granularity, single AND multiscale residual
#                       X = X_coarse + R_medium + R_fine with the partial sums supervised
#   exp2_capacity gpu 2 same architecture, unboxed: D<=384, depth<=8, batch>=8, 700 epochs
#
# Both search the backbone per trial, since the dim_head fix changes the transformer most.
set -uo pipefail
PY=/home/jakaria/anaconda3/envs/pytorch_geo/bin/python
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN=/mnt/bulk10tb/Deep3DComp/ADNI_1_LHipp/task_lamm_ae_v1/run
mkdir -p "$RUN"
[[ -d /mnt/bulk10tb/Deep3DComp ]] || { echo "FATAL: /mnt/bulk10tb is not mounted"; exit 1; }

# 550 epochs, 3200s: sized from MEASURED per-epoch cost, not guessed. A first attempt used
# 700 epochs / 2400s and truncated immediately -- bs=16/D=192 runs at 3.53 s/epoch (700 epochs
# = 2471s) and bs=8/D=384 at 8.38 s/epoch (5866s), both past a 2400s wall. --max-params bounds
# SIZE, but time is driven by step count, so batch 8 and D=384 are out of the space instead.
# Every surviving config now converges inside the budget: D=192/bs16 ~1940s,
# D=256/depth8/bs16 ~3000s. 550 epochs is 38% past v1's cap, where its median best epoch
# was 377 of 400.
COMMON=(--n-trials 14 --epochs 550 --min-epochs 80 --patience 30
        --trial-time-budget 3200 --max-params 20e6 --latent 128 --backbone search)

launch () {  # study space gpu
  local S="$1" SP="$2" G="$3"
  if [[ -f "$RUN/$S.pid" ]] && kill -0 "$(cat "$RUN/$S.pid")" 2>/dev/null; then
    echo "SKIP  $S (pid $(cat "$RUN/$S.pid"))"; return; fi
  setsid nohup "$PY" -u "$HERE/optuna_lamm.py" --study "$S" --space "$SP" --gpu "$G" \
      "${COMMON[@]}" >>"$RUN/$S.out" 2>&1 </dev/null &
  sleep 3
  pgrep -f "optuna_lamm.py --study $S " | head -1 > "$RUN/$S.pid"
  echo "START $S space=$SP gpu=$G pid=$(cat "$RUN/$S.pid")"
}

launch exp1_scales   scales 0
launch exp2_capacity v2     2
echo; echo "logs: $RUN/exp{1,2}_*.out   csv: .../studies/exp*/trial_metrics.csv"
