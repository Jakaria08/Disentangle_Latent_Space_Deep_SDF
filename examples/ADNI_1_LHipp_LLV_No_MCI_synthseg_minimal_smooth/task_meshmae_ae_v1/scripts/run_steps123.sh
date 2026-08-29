#!/usr/bin/env bash
# Steps 1-3 of the MeshMAE improvement plan, run SEQUENTIALLY on one GPU.
#
# Diagnosis these address: MeshMAE's train/val gap is 2.34x while SpiralNet++ -- with MORE
# parameters per training mesh (2676 vs 2077) -- sits at 1.57x. So the deficit is
# architectural, not capacity, and the suspect is the latent head:
#     SpiralNet++  86 verts x 192 ch = 16512  -> Linear -> 128     (keeps spatial identity)
#     MeshMAE      86 tokens x 128     -> mean+max -> 256 -> 128   (destroys it)
#
#   S1 latentset  cross-attention with 16 learned queries (3DShape2VecSet style)
#   S2 flatten    86x128 -> Linear -> 128, structurally what SpiralNet++ does
#   S3 latentset + learned locality bias from inter-patch distance (MS-SiT style)
#
# Everything else identical to config A (the current best): wd 0.05, decoder warm-started
# from spiralnet_z128_v7 and trainable, 500 epochs, per-epoch validation in mm.
set -uo pipefail
GPU="${1:-1}"
PY=/home/jakaria/anaconda3/envs/pytorch_geo/bin/python
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT=/mnt/bulk10tb/Deep3DComp/ADNI_1_LHipp/task_meshmae_ae_v1
mkdir -p "$OUT/run"

COMMON="--gpu $GPU --patch-level 3 --no-freeze-decoder --warm-start-decoder \
        --epochs 500 --eval-every 1 --min-epochs 150 --patience 25 \
        --lr 3e-4 --weight-decay 0.05 --batch-size 16"

run () {                       # $1 name, rest = extra args
  local NAME="$1"; shift
  local MARK="$OUT/run/${NAME}.done"
  if [[ -f "$MARK" ]]; then echo "[$(date +%H:%M:%S)] SKIP $NAME"; return; fi
  echo "[$(date +%H:%M:%S)] START $NAME"
  $PY -u "$HERE/train_meshmae.py" --run-name "$NAME" $COMMON "$@" \
      > "$OUT/run_${NAME}.out" 2>&1
  if [[ $? -eq 0 ]]; then touch "$MARK"; echo "[$(date +%H:%M:%S)] DONE $NAME"
  else echo "[$(date +%H:%M:%S)] FAILED $NAME -- continuing"; fi
}

run S1_latentset  --head latentset --n-queries 16
run S2_flatten    --head flatten
run S3_latentset_distbias --head latentset --n-queries 16 --distance-bias
echo "[$(date +%H:%M:%S)] queue empty"
