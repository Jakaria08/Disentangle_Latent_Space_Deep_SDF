#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 RUN_DIR PHYSICAL_GPU" >&2
  exit 2
fi

RUN_DIR="$1"
PHYSICAL_GPU="$2"
TASK="examples/ADNI_1_LHipp_LLV_No_MCI_synthseg_minimal_smooth/task3_latent_flow_128_v2_lamm"
PY="/home/jakaria/anaconda3/envs/pytorch_geo/bin/python"

export CUDA_VISIBLE_DEVICES="$PHYSICAL_GPU"
"$PY" -u "$TASK/scripts/evaluate.py" \
  --run-dir "$RUN_DIR" --split test --device cuda:0 \
  --bootstrap-samples 2000 --evaluation-name evaluation

