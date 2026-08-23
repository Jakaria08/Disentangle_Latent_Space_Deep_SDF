#!/usr/bin/env bash
set -euo pipefail

GPU="${1:-1}"
TASK="examples/ADNI_1_LHipp_LLV_No_MCI_synthseg_minimal_smooth/August_Version"
PY="/home/jakaria/anaconda3/envs/pytorch_geo/bin/python"

CUDA_VISIBLE_DEVICES="$GPU" "$PY" "$TASK/scripts/prepare_representations.py" \
  --representations pca128 spiralnet128 adaptive128 --device cuda:0 --batch-size 64
