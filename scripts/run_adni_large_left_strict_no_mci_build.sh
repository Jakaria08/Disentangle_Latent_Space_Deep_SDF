#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DATASET_ROOT="${DATASET_ROOT:-/home/jakaria/ADNI/ADNI_1_GO_Large/left_hippocampus_strict_no_mci}"
SOURCE_PLY_DIR="${SOURCE_PLY_DIR:-/home/jakaria/ADNI/ADNI_1_GO_Large/adni_hipp_ply_surfs}"
CLINICAL_CSV="${CLINICAL_CSV:-${SOURCE_PLY_DIR}/clinical_volume_merged.csv}"
REPO_EXPORT_ROOT="${REPO_EXPORT_ROOT:-${REPO_ROOT}/examples/ADNI_1_L_No_MCI_large_strict_left}"
SPLIT_EXPORT_DIR="${SPLIT_EXPORT_DIR:-${REPO_ROOT}/examples/splits/splits_left_hippocampus_ADNI_Large_Strict_No_MCI}"

INR_PYTHON="${INR_PYTHON:-/home/jakaria/anaconda3/envs/inr_sdf/bin/python}"
SHAPEWORKS_PYTHON="${SHAPEWORKS_PYTHON:-/home/jakaria/anaconda3/envs/shapeworks/bin/python}"
DEFORMETRICA_PYTHON="${DEFORMETRICA_PYTHON:-/home/jakaria/Explaining_Shape_Variability/preprocessing/deformetrica_reg/bin/python}"
PIPELINE="${REPO_ROOT}/scripts/adni_large_hippo_pipeline.py"

SEED="${SEED:-42}"
SHAPEWORKS_ITERATIONS="${SHAPEWORKS_ITERATIONS:-100}"
DEFORMETRICA_ITERATIONS="${DEFORMETRICA_ITERATIONS:-30}"
DEFORMETRICA_GPU_MODE="${DEFORMETRICA_GPU_MODE:-auto}"
TRAIN_RATIO="${TRAIN_RATIO:-0.8}"
VAL_RATIO="${VAL_RATIO:-0.1}"
TEST_RATIO="${TEST_RATIO:-0.1}"

COMMON_ARGS=(--output-root "${DATASET_ROOT}" --groups left)

"${INR_PYTHON}" "${PIPELINE}" manifest \
  "${COMMON_ARGS[@]}" \
  --source-ply-dir "${SOURCE_PLY_DIR}" \
  --clinical-csv "${CLINICAL_CSV}" \
  --sample-count 0 \
  --seed "${SEED}" \
  --diagnosis-filter strict_no_mci

"${INR_PYTHON}" "${PIPELINE}" prepare "${COMMON_ARGS[@]}"

"${SHAPEWORKS_PYTHON}" "${PIPELINE}" rigid \
  "${COMMON_ARGS[@]}" \
  --shapeworks-iterations "${SHAPEWORKS_ITERATIONS}"

"${DEFORMETRICA_PYTHON}" "${PIPELINE}" correspond \
  "${COMMON_ARGS[@]}" \
  --iterations "${DEFORMETRICA_ITERATIONS}" \
  --deformetrica-gpu-mode "${DEFORMETRICA_GPU_MODE}"

"${INR_PYTHON}" "${PIPELINE}" obj "${COMMON_ARGS[@]}"
"${INR_PYTHON}" "${PIPELINE}" labels "${COMMON_ARGS[@]}"

"${INR_PYTHON}" "${PIPELINE}" splits \
  "${COMMON_ARGS[@]}" \
  --seed "${SEED}" \
  --train-ratio "${TRAIN_RATIO}" \
  --val-ratio "${VAL_RATIO}" \
  --test-ratio "${TEST_RATIO}"

"${INR_PYTHON}" "${PIPELINE}" validate \
  "${COMMON_ARGS[@]}" \
  --skip-sdf-check

mkdir -p "${REPO_EXPORT_ROOT}/metadata" "${REPO_EXPORT_ROOT}/manifests" "${REPO_EXPORT_ROOT}/reports" "${SPLIT_EXPORT_DIR}"

LABEL_DIR="${DATASET_ROOT}/left_hippocampus_correspondence/minimal_scaled_obj_files"
cp "${LABEL_DIR}/labels.pt" "${REPO_EXPORT_ROOT}/metadata/labels.pt"
cp "${LABEL_DIR}/labels.csv" "${REPO_EXPORT_ROOT}/metadata/labels.csv"
cp "${LABEL_DIR}/label_schema.json" "${REPO_EXPORT_ROOT}/metadata/label_schema.json"
cp "${LABEL_DIR}/scale_info.json" "${REPO_EXPORT_ROOT}/metadata/scale_info.json"
cp "${DATASET_ROOT}/manifests/eligible_scans.csv" "${REPO_EXPORT_ROOT}/manifests/eligible_scans.csv"
cp "${DATASET_ROOT}/manifests/selected_scans.csv" "${REPO_EXPORT_ROOT}/manifests/selected_scans.csv"
cp "${DATASET_ROOT}/reports/"*.json "${REPO_EXPORT_ROOT}/reports/"
cp "${DATASET_ROOT}/reports/validation_details.csv" "${REPO_EXPORT_ROOT}/reports/validation_details.csv"
cp "${DATASET_ROOT}/left_hippocampus_ply_rigid_reg/reference_medoid.txt" "${REPO_EXPORT_ROOT}/reports/reference_medoid.txt"
cp "${DATASET_ROOT}/left_hippocampus_correspondence/minimal_README.txt" "${REPO_EXPORT_ROOT}/reports/correspondence_README.txt"
cp "${DATASET_ROOT}/splits/left/"*.json "${SPLIT_EXPORT_DIR}/"

echo "Build complete."
echo "Dataset root: ${DATASET_ROOT}"
echo "Repo metadata export: ${REPO_EXPORT_ROOT}"
echo "Repo split export: ${SPLIT_EXPORT_DIR}"
