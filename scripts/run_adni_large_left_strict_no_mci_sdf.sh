#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DATASET_ROOT="${DATASET_ROOT:-/home/jakaria/ADNI/ADNI_1_GO_Large/left_hippocampus_strict_no_mci}"
REPO_EXPORT_ROOT="${REPO_EXPORT_ROOT:-${REPO_ROOT}/examples/ADNI_1_L_No_MCI_large_strict_left}"
INR_PYTHON="${INR_PYTHON:-/home/jakaria/anaconda3/envs/inr_sdf/bin/python}"
PREPROCESS_SCRIPT="${PREPROCESS_SCRIPT:-${REPO_ROOT}/preprocess_data.py}"
PIPELINE="${REPO_ROOT}/scripts/adni_large_hippo_pipeline.py"
SDF_THREADS="${SDF_THREADS:-24}"
PANGOLIN_WINDOW_URI="${PANGOLIN_WINDOW_URI:-headless://}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export EGL_PLATFORM="${EGL_PLATFORM:-surfaceless}"
export LIBGL_ALWAYS_SOFTWARE="${LIBGL_ALWAYS_SOFTWARE:-1}"

COMMON_ARGS=(--output-root "${DATASET_ROOT}" --groups left)

"${INR_PYTHON}" "${PIPELINE}" sdf \
  "${COMMON_ARGS[@]}" \
  --inr-python "${INR_PYTHON}" \
  --preprocess-script "${PREPROCESS_SCRIPT}" \
  --sdf-threads "${SDF_THREADS}" \
  --pangolin-window-uri "${PANGOLIN_WINDOW_URI}"

"${INR_PYTHON}" "${PIPELINE}" validate "${COMMON_ARGS[@]}"

mkdir -p "${REPO_EXPORT_ROOT}/reports"
cp "${DATASET_ROOT}/reports/sdf_summary.json" "${REPO_EXPORT_ROOT}/reports/sdf_summary.json"
cp "${DATASET_ROOT}/reports/validation_summary.json" "${REPO_EXPORT_ROOT}/reports/validation_summary.json"
cp "${DATASET_ROOT}/reports/validation_details.csv" "${REPO_EXPORT_ROOT}/reports/validation_details.csv"

echo "SDF generation complete."
echo "Dataset root: ${DATASET_ROOT}"
echo "SDF directory: ${DATASET_ROOT}/left_hippocampus_correspondence/sdf_data/SdfSamples/minimal_scaled_obj_files"
