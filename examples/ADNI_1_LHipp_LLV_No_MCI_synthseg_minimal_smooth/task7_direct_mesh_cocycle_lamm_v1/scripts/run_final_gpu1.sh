#!/usr/bin/env bash
set -euo pipefail

TASK="/home/jakaria/INR/Deep3DComp/examples/ADNI_1_LHipp_LLV_No_MCI_synthseg_minimal_smooth/task7_direct_mesh_cocycle_lamm_v1"
PY="/home/jakaria/anaconda3/envs/inr_sdf/bin/python"

for NAME in \
  lamm_global_c4_z256_s42 \
  lamm_global_c4_z384_s42 \
  lamm_token_c4_s42
do
  "${PY}" "${TASK}/scripts/train.py" \
    --config "${TASK}/configs/${NAME}.json" \
    --device cuda:1
done
