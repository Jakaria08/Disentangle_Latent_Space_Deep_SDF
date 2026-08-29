#!/usr/bin/env bash
# Experiments 1 and 2, both backbones, seeded. 22 runs, 11 per GPU, sequential per card.
#
# E1 -- LOSS-SPACE ALIGNMENT. Training minimises L1 on (x-mu)/sigma but the reported metric is
# per-coordinate RMSE in MM, i.e. |p-t|*sigma. sigma spans 0.1223-1.2365 mm across coordinates
# (10.1x), so a coordinate enters the objective at 1/sigma relative to its weight in the
# score. Three arms:
#     N1  (x-mu)/sigma, plain L1                     -- the current convention, control
#     N2  x-mu only                                  -- what LAMM's own paper does
#     N3  (x-mu)/sigma, residual scaled by sigma     -- exactly metric-matched, verified to
#                                                       reproduce the mm-space L1 to 9.3e-8
#
# E2 -- SEED VARIANCE AND ENSEMBLE. N1 is run at 5 seeds instead of 3, so E2 reuses E1's
# control arm and costs only 4 extra runs. Gives the noise floor this project has never had:
# every ~1% claim so far (mixup 0.83%, expE 1.2%, expG 1.0%) sits on an unmeasured one.
#
# BOTH BACKBONES. The best transformer on record (0.041070) is from expD, tuned at lr<=6e-4
# and D=192, before expE/G/H/J/K found the good region; later transformer trials all used
# heads=8/dim_head=64 -> inner 512 = 2xD, which collapsed the gap to 2.4-3.4. At heads=4 the
# inner width is 256 = D, matching the mixer. So this is also the first fair transformer test
# at the tuned configuration.
set -uo pipefail
PY=/home/jakaria/anaconda3/envs/pytorch_geo/bin/python
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN=/mnt/bulk10tb/Deep3DComp/ADNI_1_LHipp/task_lamm_ae_v1/run
mkdir -p "$RUN"
[[ -d /mnt/bulk10tb/Deep3DComp ]] || { echo "FATAL: /mnt/bulk10tb is not mounted"; exit 1; }

# frozen at expK t17, the best model found (val 0.037685, 21.50M)
BASE=(--scales 43,86 --dim 256 --enc-depth 8 --dec-depth 6 --heads 4 --dim-head 64
      --batch-size 16 --lr 0.0011613991692409735 --weight-decay 2.2654123578888588e-05
      --dropout 0.05 --deep-sup 0.0 --epochs 250 --warmup-epochs 10
      --mixup-alpha 0.8 --mixup-prob 0.5 --region-mode raw --ema-decay 0.999 --loss l1
      --residual --no-subject-weight --latent 128 --patch-level 3
      --min-epochs 100 --patience 60 --eval-every 1 --time-budget 2400)

queue_for_gpu () {   # backbone short gpu
  local BB="$1" SH="$2" G="$3"
  local script="$RUN/queue_${SH}.sh"
  : > "$script"
  for seed in 1 2 3 4 5; do            # N1 gets 5 seeds (E1 control + E2 ensemble)
    echo "\"$PY\" -u \"$HERE/train_lamm.py\" --run-name E1_N1_${SH}_s${seed} --gpu $G \
--backbone $BB --seed $seed --norm-mode std --no-metric-weighted-loss ${BASE[*]}" >> "$script"
  done
  for seed in 1 2 3; do
    echo "\"$PY\" -u \"$HERE/train_lamm.py\" --run-name E1_N2_${SH}_s${seed} --gpu $G \
--backbone $BB --seed $seed --norm-mode center --no-metric-weighted-loss ${BASE[*]}" >> "$script"
    echo "\"$PY\" -u \"$HERE/train_lamm.py\" --run-name E1_N3_${SH}_s${seed} --gpu $G \
--backbone $BB --seed $seed --norm-mode std --metric-weighted-loss ${BASE[*]}" >> "$script"
  done
  setsid nohup bash -c "
    while read -r cmd; do
      echo \"[\$(date +%H:%M:%S)] START \$cmd\" ; eval \"\$cmd\" ; echo \"[\$(date +%H:%M:%S)] DONE\"
    done < '$script'
    echo '[queue] $SH finished'
  " >"$RUN/queue_${SH}.out" 2>&1 </dev/null &
  sleep 2
  pgrep -f "queue_${SH}.sh" >/dev/null 2>&1 || true
  echo "START queue $SH  backbone=$BB gpu=$G  $(wc -l < "$script") runs"
}

queue_for_gpu mlpmixer    ml 0
queue_for_gpu transformer tr 2
echo; echo "logs: $RUN/queue_{ml,tr}.out ; per-run: .../logs/E1_*.log"
