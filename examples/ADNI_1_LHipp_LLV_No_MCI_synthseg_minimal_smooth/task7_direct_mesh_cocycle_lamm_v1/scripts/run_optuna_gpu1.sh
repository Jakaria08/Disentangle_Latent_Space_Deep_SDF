#!/usr/bin/env bash
set -euo pipefail

TASK="/home/jakaria/INR/Deep3DComp/examples/ADNI_1_LHipp_LLV_No_MCI_synthseg_minimal_smooth/task7_direct_mesh_cocycle_lamm_v1"
PY="/home/jakaria/anaconda3/envs/inr_sdf/bin/python"

"${PY}" "${TASK}/scripts/validate_experiment.py" --device cuda:1
"${PY}" "${TASK}/scripts/optuna_search.py" \
  --device cuda:1 \
  --n-trials "${1:-18}" \
  --trial-epochs "${2:-80}" \
  --trial-samples-per-epoch "${3:-384}" \
  --study-tag main_v1

