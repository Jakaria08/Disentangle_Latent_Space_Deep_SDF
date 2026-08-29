#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "usage: $0 REPRESENTATION PHYSICAL_GPU [SEED]" >&2
  exit 2
fi

REPRESENTATION="$1"
PHYSICAL_GPU="$2"
SEED="${3:-42}"
TASK="examples/ADNI_1_LHipp_LLV_No_MCI_synthseg_minimal_smooth/task3_latent_flow_128_v2_lamm"
ROOT="/mnt/bulk10tb/Deep3DComp/ADNI_1_LHipp/task3_latent_flow_128_v3_lamm_latest"
PY="/home/jakaria/anaconda3/envs/pytorch_geo/bin/python"

case "$REPRESENTATION" in
  spiralnet128|lamm128|lamm128_n3_s2|lamm128_n3_s3) ;;
  *) echo "unsupported representation: $REPRESENTATION" >&2; exit 2 ;;
esac

CONFIG="$TASK/configs/${REPRESENTATION}_direct_c4_s42.json"
RUN_NAME="${REPRESENTATION}_direct_c4_s${SEED}"
RUN_DIR="$ROOT/training/$REPRESENTATION/direct_c4/$RUN_NAME"

export CUDA_VISIBLE_DEVICES="$PHYSICAL_GPU"
"$PY" -u "$TASK/scripts/train_c4.py" \
  --config "$CONFIG" --device cuda:0 --seed "$SEED" --run-name "$RUN_NAME"

# Validation is safe for checkpoint/seed selection.  Test remains sealed.
"$PY" -u "$TASK/scripts/evaluate.py" \
  --run-dir "$RUN_DIR" --split val --device cuda:0 \
  --bootstrap-samples 2000 --evaluation-name evaluation
