#!/usr/bin/env bash
set -euo pipefail

GPU="${1:-1}"
TASK="examples/ADNI_1_LHipp_LLV_No_MCI_synthseg_minimal_smooth/task3_latent_flow_128_v2_lamm"
PY="/home/jakaria/anaconda3/envs/pytorch_geo/bin/python"

CUDA_VISIBLE_DEVICES="$GPU" "$PY" "$TASK/scripts/prepare_representations.py" \
  --representations lamm128_n3_s2 lamm128_n3_s3 --device cuda:0 --batch-size 32
