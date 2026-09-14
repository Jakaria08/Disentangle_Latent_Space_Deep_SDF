#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 PHYSICAL_GPU [SEED]" >&2
  exit 2
fi

PHYSICAL_GPU="$1"
SEED="${2:-42}"
TASK="examples/ADNI_1_LHipp_LLV_No_MCI_synthseg_minimal_smooth/task3_latent_flow_128_v1"
PY="/home/jakaria/anaconda3/envs/pytorch_geo/bin/python"
CONFIG="$TASK/configs/pca128_brainode_cognition_s42.json"
OUTPUT_ROOT="/mnt/bulk10tb/Deep3DComp/ADNI_1_LHipp/task3_latent_flow_128_v1"
RUN_NAME="pca128_brainode_cognition_s${SEED}"
RUN_DIR="$OUTPUT_ROOT/training/pca128/brainode_cognition/$RUN_NAME"
COCYCLE_RUN="$OUTPUT_ROOT/training/pca128/direct_c4/pca128_direct_c4_s42"
VOXEL_MANIFEST="$OUTPUT_ROOT/cognition_voxels/pca128/voxel32/manifest.json"

if [[ ! -f "$VOXEL_MANIFEST" ]]; then
  "$PY" -u "$TASK/scripts/prepare_cognition_voxels.py" \
    --resolution 32 --padding-voxels 2 --workers 8
fi

export CUDA_VISIBLE_DEVICES="$PHYSICAL_GPU"
if [[ ! -f "$RUN_DIR/checkpoints/combined_best.pt" ]]; then
  TRAIN_ARGS=(--config "$CONFIG" --device cuda:0 --seed "$SEED" --run-name "$RUN_NAME" --stage all)
  if [[ -d "$RUN_DIR" ]]; then TRAIN_ARGS+=(--resume); fi
  "$PY" -u "$TASK/scripts/train_brainode_cognition.py" "${TRAIN_ARGS[@]}"
fi

for SPLIT in val test; do
  if [[ ! -f "$RUN_DIR/brainode_cognition_evaluation/$SPLIT/summary.json" ]]; then
    "$PY" -u "$TASK/scripts/evaluate_brainode_cognition.py" \
      --run-dir "$RUN_DIR" --split "$SPLIT" --device cuda:0 \
      --bootstrap-samples 2000 --evaluation-name brainode_cognition_evaluation
  fi
  if [[ ! -f "$COCYCLE_RUN/brainode_cognition_comparator/$SPLIT/summary.json" ]]; then
    "$PY" -u "$TASK/scripts/evaluate.py" \
      --run-dir "$COCYCLE_RUN" --split "$SPLIT" --device cuda:0 \
      --bootstrap-samples 2000 --evaluation-name brainode_cognition_comparator
  fi
  COMPARISON="$RUN_DIR/comparison/${SPLIT}_brainode_cognition_vs_pca_cocycle.csv"
  if [[ ! -f "$COMPARISON" ]]; then
    "$PY" -u "$TASK/scripts/compare_brainode_cognition_cocycle.py" \
      --brainode-run "$RUN_DIR" --cocycle-run "$COCYCLE_RUN" --split "$SPLIT" \
      --bootstrap-samples 5000 --output "$COMPARISON"
  fi
done

echo "COMPLETE: $RUN_DIR"
