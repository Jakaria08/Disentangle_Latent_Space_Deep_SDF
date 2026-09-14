#!/usr/bin/env bash
# Stage 1 (data foundation), run end to end as one detached job that survives SSH loss:
#
#   setsid nohup bash scripts/run_stage1_detached.sh > /dev/null 2>&1 &
#
# Steps run in order and the script stops at the first failure. Every step logs to
# $BULK/logs/stage1_<step>.log and the overall state goes to $BULK/logs/stage1_status.txt.
# GPU work uses cuda:2 only (GPU 1 stays free; GPU 0 may be busy with other jobs).
set -euo pipefail

TASK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BULK="/mnt/bulk10tb/Deep3DComp/LHipp_LatentDynamics_CrossCohort_v1/stage1_data_foundation"
GPU_PY="/home/jakaria/anaconda3/envs/pytorch_geo/bin/python"
TEST_PY="/home/jakaria/anaconda3/envs/inr_sdf/bin/python"
DEVICE="${STAGE1_DEVICE:-cuda:2}"
OVERWRITE="${STAGE1_OVERWRITE:-}"   # set to --overwrite to rebuild existing outputs

case "$DEVICE" in cuda:1|cuda) echo "refusing device $DEVICE: use cuda:0 or cuda:2" >&2; exit 2;; esac
mkdir -p "$BULK/logs"
STATUS="$BULK/logs/stage1_status.txt"
cd "$TASK_ROOT/scripts"

step() {
  local name="$1"; shift
  echo "$(date '+%F %T') START $name" >> "$STATUS"
  if "$@" > "$BULK/logs/stage1_${name}.log" 2>&1; then
    echo "$(date '+%F %T') OK    $name" >> "$STATUS"
  else
    local rc=$?
    echo "$(date '+%F %T') FAIL  $name (exit $rc) - see $BULK/logs/stage1_${name}.log" >> "$STATUS"
    exit "$rc"
  fi
}

echo "$(date '+%F %T') stage 1 begin (pid $$, device $DEVICE)" >> "$STATUS"
step tests               "$TEST_PY" "$TASK_ROOT/tests/test_stage1_data_foundation.py"
step inclusive_cohorts   "$GPU_PY" stage1_build_inclusive_cohorts.py $OVERWRITE
step encode_latents      "$GPU_PY" stage1_encode_cohort_latents.py --device "$DEVICE" $OVERWRITE
step protocol_views      "$GPU_PY" stage1_build_protocol_views.py $OVERWRITE
step baselines           "$GPU_PY" stage1_compute_baselines.py --device "$DEVICE" --splits val test
step audit               "$GPU_PY" stage1_audit_data_foundation.py
echo "$(date '+%F %T') stage 1 complete" >> "$STATUS"
