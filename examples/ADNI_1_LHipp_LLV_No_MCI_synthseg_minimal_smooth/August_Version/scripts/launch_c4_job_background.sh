#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 5 || $# -gt 6 ]]; then
  echo "usage: $0 CONFIG REPRESENTATION PHYSICAL_GPU SEED RUN_NAME [JOB_NAME]" >&2
  exit 2
fi

CONFIG="$1"
REPRESENTATION="$2"
PHYSICAL_GPU="$3"
SEED="$4"
RUN_NAME="$5"
JOB_NAME="${6:-$RUN_NAME}"
TASK="examples/ADNI_1_LHipp_LLV_No_MCI_synthseg_minimal_smooth/August_Version"
OUTPUT_ROOT="/mnt/bulk10tb/Deep3DComp/ADNI_1_LHipp/August_Version"
JOB_DIR="$OUTPUT_ROOT/jobs/$JOB_NAME"
RUN_DIR="$OUTPUT_ROOT/training/$REPRESENTATION/direct_c4/$RUN_NAME"
SESSION_NAME="task3_${JOB_NAME}"

if [[ -e "$JOB_DIR" || -e "$RUN_DIR" ]]; then
  echo "refusing to overwrite existing job or run: $JOB_DIR / $RUN_DIR" >&2
  exit 1
fi
mkdir -p "$JOB_DIR"
screen -L -Logfile "$JOB_DIR/job.log" -dmS "$SESSION_NAME" \
  "$TASK/scripts/run_c4_job_gpu.sh" \
  "$CONFIG" "$REPRESENTATION" "$PHYSICAL_GPU" "$SEED" "$RUN_NAME" "$JOB_DIR"
printf '%s\n' "$SESSION_NAME" > "$JOB_DIR/screen.session"
sleep 1
if [[ ! -f "$JOB_DIR/status.json" ]]; then
  echo "background job failed to remain alive; inspect $JOB_DIR/job.log" >&2
  exit 1
fi
RUNNER_PID=$(sed -n 's/.*"pid": \([0-9][0-9]*\).*/\1/p' "$JOB_DIR/status.json")
printf '%s\n' "$RUNNER_PID" > "$JOB_DIR/job.pid"
printf 'JOB_NAME=%s\n' "$JOB_NAME"
printf 'SESSION=%s\n' "$SESSION_NAME"
printf 'RUNNER_PID=%s\n' "$RUNNER_PID"
