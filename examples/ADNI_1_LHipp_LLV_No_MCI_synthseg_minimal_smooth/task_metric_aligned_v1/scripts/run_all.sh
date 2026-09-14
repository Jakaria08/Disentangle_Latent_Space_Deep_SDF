#!/usr/bin/env bash
# Metric-aligned experiments A, B and C. NEW TREE -- nothing here writes into
# task_lamm_ae_v1 or task_spiral_ae_v1; every published study and checkpoint there is left
# untouched.
#
# THE CORRECTION UNDER TEST. Every model in this project trains L1 on (x-mu)/sigma but is
# scored on per-coordinate RMSE in MILLIMETRES, i.e. |p-t|*sigma. sigma spans 0.1223-1.2365 mm
# (10.1x), so a coordinate enters the objective at 1/sigma relative to its weight in the
# score. Fixing it on LAMM (E1) was worth 10.8%:
#     N1  sigma-normalised, plain L1 (old)   0.038051 +- 0.000256  (5 seeds)
#     N2  mean-centred only                  0.034020 +- 0.000119  (3 seeds)
#     N3  sigma-weighted loss                0.033916 +- 0.000131  (3 seeds)  <- 0.74% from PCA
# against a measured seed noise floor of 0.77%, so ~16 sigma.
#
#   A  gpu 0  re-tune architecture + hyperparameters under the corrected objective. All 216
#             previous trials optimised the wrong target, so no verdict from them survives.
#   B  gpu 1  loss function under the corrected objective. sigma-weighted L2 is EXACTLY the
#             metric squared -- mean(((p-t)*sigma)^2) -- while N3's sigma-weighted L1 matches
#             the mm mean-ABSOLUTE error. L1 beat L2 earlier (expI) but that test ran under
#             the sigma distortion, where L2 amplifies it (1/sigma^2 vs 1/sigma).
#   C  gpu 2  SpiralNet++ with the same fix. Its 0.036784 was produced under the defect and
#             has never been re-run, so the comparison table is untrustworthy in BOTH
#             directions until this lands. PCA is unaffected -- it has no training objective.
set -uo pipefail
PY=/home/jakaria/anaconda3/envs/pytorch_geo/bin/python
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAMM="$HERE/../../task_lamm_ae_v1/scripts"
ROOT=/mnt/bulk10tb/Deep3DComp/ADNI_1_LHipp/task_metric_aligned_v1
RUN="$ROOT/run"; mkdir -p "$RUN"
[[ -d /mnt/bulk10tb/Deep3DComp ]] || { echo "FATAL: /mnt/bulk10tb is not mounted"; exit 1; }

# frozen at E1 arm N3 / mlpmixer, the best model on record (0.033916 +- 0.000131)
BASE=(--scales 43,86 --dim 256 --enc-depth 8 --dec-depth 6 --heads 4 --dim-head 64
      --batch-size 16 --lr 0.0011613991692409735 --weight-decay 2.2654123578888588e-05
      --dropout 0.05 --deep-sup 0.0 --epochs 250 --warmup-epochs 10
      --mixup-alpha 0.8 --mixup-prob 0.5 --region-mode raw --ema-decay 0.999
      --residual --no-subject-weight --latent 128 --patch-level 3
      --min-epochs 100 --patience 60 --eval-every 1 --time-budget 2400
      --norm-mode std --metric-weighted-loss --backbone mlpmixer)

# ---------------- A: re-tune under the corrected objective (gpu 0) ----------------
setsid nohup "$PY" -u "$LAMM/optuna_lamm.py" --study A_retune_aligned --space aligned \
    --backbone mlpmixer --gpu 0 --out-root "$ROOT" --n-trials 24 --min-epochs 100 \
    --patience 60 --trial-time-budget 2400 --max-params 36e6 --latent 128 \
    >>"$RUN/A_retune_aligned.out" 2>&1 </dev/null &
sleep 3; pgrep -f "optuna_lamm.py --study A_retune_aligned" | while read p; do
  [[ "$(ps -o ppid= -p $p|tr -d ' ')" == "1" ]] && echo $p > "$RUN/A_retune_aligned.pid"; done
echo "START A_retune_aligned  gpu=0  24 trials  pid=$(cat $RUN/A_retune_aligned.pid)"

# ---------------- B: loss under the corrected objective (gpu 1) ----------------
: > "$RUN/queue_B.sh"
for loss in l1 l2 huber; do for seed in 1 2 3; do
  echo "\"$PY\" -u \"$LAMM/train_lamm.py\" --run-name B_${loss}_s${seed} --gpu 1 \
--out-root '$ROOT' --seed $seed --loss $loss ${BASE[*]}" >> "$RUN/queue_B.sh"
done; done
setsid nohup bash -c "while read -r c; do echo \"[\$(date +%H:%M:%S)] START \$c\"; eval \"\$c\";
  done < '$RUN/queue_B.sh'; echo '[queue B] finished'" >"$RUN/queue_B.out" 2>&1 </dev/null &
echo "START queue B  gpu=1  $(wc -l < "$RUN/queue_B.sh") runs (l1/l2/huber x 3 seeds)"

# ---------------- C: SpiralNet++ with the same fix (gpu 2) ----------------
: > "$RUN/queue_C.sh"
for arm in "N1:--norm-mode std --no-metric-weighted-loss" \
           "N2:--norm-mode center --no-metric-weighted-loss" \
           "N3:--norm-mode std --metric-weighted-loss"; do
  IFS=: read -r NM FL <<<"$arm"
  for seed in 1 2 3; do
    echo "\"$PY\" -u \"$HERE/train_spiral_aligned.py\" --run-name C_${NM}_sp_s${seed} --gpu 2 \
--out-root '$ROOT' --seed $seed --conv-type spiral --eval-every 5 --time-budget 2400 $FL" \
      >> "$RUN/queue_C.sh"
  done
done
setsid nohup bash -c "while read -r c; do echo \"[\$(date +%H:%M:%S)] START \$c\"; eval \"\$c\";
  done < '$RUN/queue_C.sh'; echo '[queue C] finished'" >"$RUN/queue_C.out" 2>&1 </dev/null &
echo "START queue C  gpu=2  $(wc -l < "$RUN/queue_C.sh") runs (N1/N2/N3 x 3 seeds)"
echo; echo "outputs: $ROOT   logs: $RUN/"
