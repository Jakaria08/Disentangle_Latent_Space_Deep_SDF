#!/usr/bin/env bash
set -euo pipefail

TASK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="$(cd "${TASK_ROOT}/../../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/jakaria/anaconda3/envs/pytorch3d/bin/python}"
BULK_ROOT="/mnt/bulk10tb/Deep3DComp/ADNI_1_LHipp/August_Version"
STAMP="$(date +%Y%m%d_%H%M%S)"
LAUNCH_DIR="${BULK_ROOT}/launches/volume_exact_coboundary_v2_s42_${STAMP}"
representations=(pca128 spiralnet128 adaptive128)
pids=()

cleanup() {
  local pid
  for pid in "${pids[@]:-}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" 2>/dev/null || true
    fi
  done
}
trap cleanup INT TERM EXIT

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python executable is unavailable: ${PYTHON_BIN}" >&2
  exit 2
fi
if ! nvidia-smi -i 2 --query-gpu=index,memory.used,memory.total --format=csv,noheader; then
  echo "Physical GPU 2 is unavailable; no experiment was started." >&2
  exit 3
fi
active_gpu2="$(nvidia-smi -i 2 --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null || true)"
if [[ -n "${active_gpu2}" ]]; then
  echo "Physical GPU 2 already has compute processes; refusing to interfere:" >&2
  echo "${active_gpu2}" >&2
  exit 4
fi

for representation in "${representations[@]}"; do
  run_dir="${BULK_ROOT}/training/${representation}/volume_exact_coboundary_c4_v2/${representation}_volume_exact_coboundary_v2_s42"
  if [[ -e "${run_dir}" ]]; then
    echo "Refusing to overwrite existing target: ${run_dir}" >&2
    exit 5
  fi
done

mkdir -p "${LAUNCH_DIR}"
export CUDA_VISIBLE_DEVICES=2
export PHYSICAL_GPU_ID=2
cd "${PROJECT_ROOT}"

for representation in "${representations[@]}"; do
  config="${TASK_ROOT}/configs/${representation}_volume_exact_coboundary_v2_s42.json"
  log="${LAUNCH_DIR}/${representation}_train.log"
  "${PYTHON_BIN}" "${TASK_ROOT}/scripts/train_volume_coboundary.py" \
    --config "${config}" --device cuda:0 >"${log}" 2>&1 &
  pids+=("$!")
  echo "started ${representation} pid=${pids[-1]} log=${log}"
done

failed=0
for index in "${!pids[@]}"; do
  if wait "${pids[$index]}"; then
    echo "training complete ${representations[$index]}"
  else
    echo "training failed ${representations[$index]}" >&2
    failed=1
  fi
done
pids=()
if [[ "${failed}" -ne 0 ]]; then
  echo "At least one training process failed; evaluation was not started." >&2
  exit 6
fi

for representation in "${representations[@]}"; do
  run_dir="${BULK_ROOT}/training/${representation}/volume_exact_coboundary_c4_v2/${representation}_volume_exact_coboundary_v2_s42"
  for split in val test; do
    "${PYTHON_BIN}" "${TASK_ROOT}/scripts/evaluate_volume_coboundary.py" \
      --run-dir "${run_dir}" --split "${split}" --device cuda:0 \
      --evaluation-name evaluation >"${LAUNCH_DIR}/${representation}_evaluate_${split}.log" 2>&1
  done
done

comparison_dir="${BULK_ROOT}/comparisons/volume_exact_coboundary_v2_s42_${STAMP}"
"${PYTHON_BIN}" "${TASK_ROOT}/scripts/compare_volume_coboundary.py" \
  --output-dir "${comparison_dir}" >"${LAUNCH_DIR}/comparison.log" 2>&1
touch "${LAUNCH_DIR}/COMPLETE"
echo "COMPLETE launch_dir=${LAUNCH_DIR} comparison_dir=${comparison_dir}"
