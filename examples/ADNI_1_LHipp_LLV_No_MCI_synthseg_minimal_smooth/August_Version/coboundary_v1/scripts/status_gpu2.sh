#!/usr/bin/env bash
set -euo pipefail

BULK_ROOT="/mnt/bulk10tb/Deep3DComp/ADNI_1_LHipp/August_Version"
TRAINING_ROOT="${BULK_ROOT}/training"
PYTHON_BIN="${PYTHON_BIN:-/home/jakaria/anaconda3/envs/pytorch3d/bin/python}"

screen -ls 2>/dev/null | grep -F "exact_cob_s42_gpu2" || true
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader

for representation in pca128 spiralnet128 adaptive128; do
  run_dir="${TRAINING_ROOT}/${representation}/exact_coboundary_c4/${representation}_exact_coboundary_c4_s42"
  status_path="${run_dir}/training_status.json"
  if [[ -f "${status_path}" ]]; then
    "${PYTHON_BIN}" -c '
import json, sys
d = json.load(open(sys.argv[1]))
estimate = d.get("estimated_total_minutes_at_current_rate")
remaining = None if estimate is None else max(float(estimate) - float(d.get("elapsed_minutes", 0.0)), 0.0)
print(f"{sys.argv[2]:12s} status={d.get(chr(115)+chr(116)+chr(97)+chr(116)+chr(117)+chr(115))} "
      f"epoch={d.get(chr(101)+chr(112)+chr(111)+chr(99)+chr(104))}/{d.get(chr(101)+chr(112)+chr(111)+chr(99)+chr(104)+chr(115)+chr(95)+chr(114)+chr(101)+chr(113)+chr(117)+chr(101)+chr(115)+chr(116)+chr(101)+chr(100))} "
      f"best={d.get(chr(98)+chr(101)+chr(115)+chr(116)+chr(95)+chr(118)+chr(97)+chr(108)+chr(105)+chr(100)+chr(97)+chr(116)+chr(105)+chr(111)+chr(110)+chr(95)+chr(115)+chr(101)+chr(108)+chr(101)+chr(99)+chr(116)+chr(105)+chr(111)+chr(110)+chr(95)+chr(115)+chr(99)+chr(111)+chr(114)+chr(101))} "
      f"elapsed_min={float(d.get(chr(101)+chr(108)+chr(97)+chr(112)+chr(115)+chr(101)+chr(100)+chr(95)+chr(109)+chr(105)+chr(110)+chr(117)+chr(116)+chr(101)+chr(115), 0.0)):.1f} "
      f"projected_remaining_min={remaining if remaining is not None else chr(110)+chr(47)+chr(97)}")
' "${status_path}" "${representation}"
  else
    printf '%-12s status=initializing\n' "${representation}"
  fi
done

latest_launch="$(find "${BULK_ROOT}/launches" -maxdepth 1 -type d -name 'exact_coboundary_c4_s42_*' | sort | tail -1)"
if [[ -n "${latest_launch}" ]]; then
  echo "launch_dir=${latest_launch}"
fi

