#!/usr/bin/env bash
# Phase 2: semi-amortised LAMM (two arms) + Adaptive Spiral with the objective fix.
#
# CONTEXT. Aligning the training objective with the metric collapsed the field: PCA 0.033668,
# LAMM huber 0.033665, LAMM L2 0.033698, SpiralNet++ fixed 0.033881 -- all within 0.6%, with
# seed noise +-0.47%. Two things remain.
#
#   ARM 1 (gpu 1)  post-hoc latent refinement. PCA's encoder z = U^T(x-mu) is the EXACT
#                  least-squares argmin for its decoder, so its amortisation gap is zero by
#                  construction; a learned encoder only approximates its own decoder's optimum.
#                  Measured on B_huber_s1: 1.09% val, 1.08% test at 200 steps (0.40% at 50),
#                  so the step count is not yet saturated -- hence the sweep. Refinement starts
#                  from the encoder's own output and descends the same objective, so it is
#                  non-increasing per sample: this arm cannot make anything worse.
#   ARM 2 (gpu 0)  semi-amortised TRAINING. K inner steps refine z, then a second decoder loss
#                  on the refined code, so the decoder is fitted to the codes it will actually
#                  be given. Measured cost 1.6x/2.3x/3.1x baseline for K=1/3/5, so the 4000s
#                  budget covers K=5's ~3325s at 250 epochs and nothing is truncated -- K is
#                  the variable under test and must not be biased by the wall.
#   TASK 3 (gpu 2) Adaptive Spiral with the same objective fix. Its published 0.037237 was
#                  produced under the defect. Arm N1 reproduces adaptive_z128_v7 trial_0023
#                  exactly (conv_types [spiral,spiral,adaptive], dyn_seq [1,1,187], verified)
#                  and acts as the control.
set -uo pipefail
PY=/home/jakaria/anaconda3/envs/pytorch_geo/bin/python
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAMM="$HERE/../../task_lamm_ae_v1/scripts"
ROOT=/mnt/bulk10tb/Deep3DComp/ADNI_1_LHipp/task_metric_aligned_v1
RUN="$ROOT/run"; mkdir -p "$RUN"
[[ -d /mnt/bulk10tb/Deep3DComp ]] || { echo "FATAL: /mnt/bulk10tb is not mounted"; exit 1; }

start_queue () {  # name script
  setsid nohup bash -c "while read -r c; do echo \"[\$(date +%H:%M:%S)] START \$c\"; eval \"\$c\";
    done < '$2'; echo '[queue $1] finished'" >"$RUN/queue_$1.out" 2>&1 </dev/null &
  echo "START queue $1  $(wc -l < "$2") jobs"
}

# ---------------- ARM 2: semi-amortised training (gpu 0) ----------------
BASE="--scales 43,86 --dim 256 --enc-depth 8 --dec-depth 6 --heads 4 --dim-head 64 \
--batch-size 16 --lr 0.0011613991692409735 --weight-decay 2.2654123578888588e-05 \
--dropout 0.05 --deep-sup 0.0 --epochs 250 --warmup-epochs 10 --mixup-alpha 0.8 \
--mixup-prob 0.5 --region-mode raw --ema-decay 0.999 --residual --no-subject-weight \
--latent 128 --patch-level 3 --min-epochs 100 --patience 60 --eval-every 1 \
--norm-mode std --metric-weighted-loss --loss huber --backbone mlpmixer --time-budget 4000"
: > "$RUN/q_SA.sh"
for K in 1 3 5; do for s in 1 2 3; do
  echo "\"$PY\" -u \"$LAMM/train_lamm.py\" --run-name SA_K${K}_s${s} --gpu 0 \
--out-root '$ROOT' --seed $s --semi-amortized $K --sa-lr 3e-3 --sa-weight 1.0 $BASE" >> "$RUN/q_SA.sh"
done; done
start_queue SA "$RUN/q_SA.sh"

# ---------------- TASK 3: Adaptive Spiral, objective fix (gpu 2) ----------------
# adaptive_z128_v7 trial_0023 (val 0.037237): seq 13, dil 2, base 96, dropout 0.1,
# lr 1.828e-4, decay 0.9258/8, wd 7.0117e-5, noise 0.06, adaptive on the coarsest level.
AD="--conv-type adaptive --seq-length 13 --dilation 2 --base-channels 96 --dropout 0.1 \
--lr 0.00018278378569236954 --lr-decay 0.9258173117763585 --decay-step 8 \
--weight-decay 7.011680125038927e-05 --noise-std 0.06 --adaptive-levels 1 \
--dyn-frac 0.5431357742843035 --epochs 400 --eval-every 5 --time-budget 3000"
: > "$RUN/q_AD.sh"
for arm in "N1:--norm-mode std --no-metric-weighted-loss" \
           "N2:--norm-mode center --no-metric-weighted-loss" \
           "N3:--norm-mode std --metric-weighted-loss"; do
  IFS=: read -r NM FL <<<"$arm"
  for s in 1 2 3; do
    echo "\"$PY\" -u \"$HERE/train_spiral_aligned.py\" --run-name D_${NM}_ad_s${s} --gpu 2 \
--out-root '$ROOT' --seed $s $AD $FL" >> "$RUN/q_AD.sh"
  done
done
start_queue AD "$RUN/q_AD.sh"

# ---------------- ARM 1: post-hoc refinement sweep (gpu 1) ----------------
: > "$RUN/q_RF.sh"
echo "\"$PY\" -u \"$HERE/refine_eval.py\" --gpu 1 --root '$ROOT' --label refine_lamm_huber \
--runs B_huber_s1 B_huber_s2 B_huber_s3 --steps 200 1000 --lrs 3e-3 1e-2" >> "$RUN/q_RF.sh"
echo "\"$PY\" -u \"$HERE/refine_eval.py\" --gpu 1 --root '$ROOT' --label refine_lamm_l2 \
--runs B_l2_s1 B_l2_s2 B_l2_s3 --steps 200 1000 --lrs 3e-3 1e-2" >> "$RUN/q_RF.sh"
echo "\"$PY\" -u \"$HERE/refine_eval.py\" --gpu 1 --root '$ROOT' --label refine_spiral_fixed \
--runs C_N3_sp_s1 C_N3_sp_s2 C_N3_sp_s3 --steps 200 1000 --lrs 3e-3 1e-2" >> "$RUN/q_RF.sh"
start_queue RF "$RUN/q_RF.sh"

echo; echo "outputs: $ROOT   logs: $RUN/queue_{SA,AD,RF}.out"
