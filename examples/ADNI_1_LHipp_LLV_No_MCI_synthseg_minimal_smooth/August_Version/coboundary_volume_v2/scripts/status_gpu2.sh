#!/usr/bin/env bash
set -euo pipefail

BULK_ROOT="/mnt/bulk10tb/Deep3DComp/ADNI_1_LHipp/August_Version"
TRAINING_ROOT="${BULK_ROOT}/training"
PYTHON_BIN="${PYTHON_BIN:-/home/jakaria/anaconda3/envs/pytorch3d/bin/python}"

screen -ls 2>/dev/null | grep -F "volume_cob_v2_s42_gpu2" || true
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader

for representation in pca128 spiralnet128 adaptive128; do
  run_dir="${TRAINING_ROOT}/${representation}/volume_exact_coboundary_c4_v2/${representation}_volume_exact_coboundary_v2_s42"
  status_path="${run_dir}/training_status.json"
  if [[ -f "${status_path}" ]]; then
    "${PYTHON_BIN}" -c '
import json, sys
d = json.load(open(sys.argv[1], encoding="utf-8"))
estimate = d.get("estimated_total_minutes_at_current_rate")
elapsed = float(d.get("elapsed_minutes", 0.0))
remaining = "n/a" if estimate is None else f"{max(float(estimate)-elapsed, 0.0):.1f}"
print(f"{sys.argv[2]:12s} status={d.get('"'"'status'"'"')} epoch={d.get('"'"'epoch'"'"')}/{d.get('"'"'epochs_requested'"'"')} "
      f"best={d.get('"'"'best_validation_selection_score'"'"')}@{d.get('"'"'best_epoch'"'"')} "
      f"elapsed_min={elapsed:.1f} projected_remaining_min={remaining} "
      f"CN={d.get('"'"'current_cn_predicted_signed_rate'"'"')} AD={d.get('"'"'current_ad_predicted_signed_rate'"'"')}")
' "${status_path}" "${representation}"
  else
    printf '%-12s status=initializing\n' "${representation}"
  fi
done

latest_launch="$(find "${BULK_ROOT}/launches" -maxdepth 1 -type d -name 'volume_exact_coboundary_v2_s42_*' | sort | tail -1)"
if [[ -n "${latest_launch}" ]]; then
  echo "launch_dir=${latest_launch}"
fi
