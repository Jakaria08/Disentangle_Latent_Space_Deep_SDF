#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
task_dir="$(cd "${script_dir}/.." && pwd)"

bash "${script_dir}/run_on_bulk.sh" -m unittest -v \
    "${task_dir}/tests/test_sampling_ablation.py"

bash "${script_dir}/run_on_bulk.sh" \
    "${script_dir}/validate_sampling_ablation.py" \
    --require-data \
    --sampling-scans 3
