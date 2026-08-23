#!/usr/bin/env bash
set -euo pipefail

PHYSICAL_GPU="${1:-1}"
SEED="${2:-42}"
LAUNCH_TAG="${3:-screen1}"
TASK="examples/ADNI_1_LHipp_LLV_No_MCI_synthseg_minimal_smooth/task3_latent_flow_128_v1"
OUTPUT_ROOT="/mnt/bulk10tb/Deep3DComp/ADNI_1_LHipp/task3_latent_flow_128_v1"
PCA_RUN_NAME="pca128_direct_c4_s${SEED}"
SPIRAL_RUN_NAME="spiralnet128_direct_c4_s${SEED}"
PCA_RUN_DIR="$OUTPUT_ROOT/training/pca128/direct_c4/$PCA_RUN_NAME"
SPIRAL_RUN_DIR="$OUTPUT_ROOT/training/spiralnet128/direct_c4/$SPIRAL_RUN_NAME"
PCA_JOB_NAME="${PCA_RUN_NAME}_${LAUNCH_TAG}"
SPIRAL_JOB_NAME="${SPIRAL_RUN_NAME}_${LAUNCH_TAG}"
PCA_JOB_DIR="$OUTPUT_ROOT/jobs/$PCA_JOB_NAME"
SPIRAL_JOB_DIR="$OUTPUT_ROOT/jobs/$SPIRAL_JOB_NAME"
COMPARE_JOB_NAME="compare_pca_spiral_direct_c4_s${SEED}_${LAUNCH_TAG}"
COMPARE_JOB_DIR="$OUTPUT_ROOT/jobs/$COMPARE_JOB_NAME"
COMPARE_SESSION="task3_${COMPARE_JOB_NAME}"

if [[ -e "$COMPARE_JOB_DIR" ]]; then
  echo "refusing to overwrite comparison job: $COMPARE_JOB_DIR" >&2
  exit 1
fi

"$TASK/scripts/launch_c4_job_background.sh" \
  "$TASK/configs/pca128_direct_c4_s42.json" pca128 "$PHYSICAL_GPU" "$SEED" "$PCA_RUN_NAME" "$PCA_JOB_NAME"
"$TASK/scripts/launch_c4_job_background.sh" \
  "$TASK/configs/spiralnet128_direct_c4_s42.json" spiralnet128 "$PHYSICAL_GPU" "$SEED" "$SPIRAL_RUN_NAME" "$SPIRAL_JOB_NAME"

mkdir -p "$COMPARE_JOB_DIR"
screen -L -Logfile "$COMPARE_JOB_DIR/job.log" -dmS "$COMPARE_SESSION" \
  "$TASK/scripts/run_two_c4_comparison_when_done.sh" \
  "$PCA_JOB_DIR" "$SPIRAL_JOB_DIR" "$PCA_RUN_DIR" "$SPIRAL_RUN_DIR" "$SEED" "$COMPARE_JOB_DIR" "$OUTPUT_ROOT"
printf '%s\n' "$COMPARE_SESSION" > "$COMPARE_JOB_DIR/screen.session"
sleep 1
COMPARE_PID=$(sed -n 's/.*"pid": \([0-9][0-9]*\).*/\1/p' "$COMPARE_JOB_DIR/status.json")
printf '%s\n' "$COMPARE_PID" > "$COMPARE_JOB_DIR/job.pid"

printf 'COMPARE_PID=%s\n' "$COMPARE_PID"
printf 'PCA_LOG=%s\n' "$PCA_JOB_DIR/job.log"
printf 'SPIRAL_LOG=%s\n' "$SPIRAL_JOB_DIR/job.log"
printf 'COMPARE_LOG=%s\n' "$COMPARE_JOB_DIR/job.log"
