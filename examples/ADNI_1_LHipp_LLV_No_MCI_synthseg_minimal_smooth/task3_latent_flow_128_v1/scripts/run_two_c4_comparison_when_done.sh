#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 7 ]]; then
  echo "usage: $0 PCA_JOB_DIR SPIRAL_JOB_DIR PCA_RUN_DIR SPIRAL_RUN_DIR SEED JOB_DIR OUTPUT_ROOT" >&2
  exit 2
fi

PCA_JOB_DIR="$1"
SPIRAL_JOB_DIR="$2"
PCA_RUN_DIR="$3"
SPIRAL_RUN_DIR="$4"
SEED="$5"
JOB_DIR="$6"
OUTPUT_ROOT="$7"
TASK="examples/ADNI_1_LHipp_LLV_No_MCI_synthseg_minimal_smooth/task3_latent_flow_128_v1"
PY="/home/jakaria/anaconda3/envs/pytorch_geo/bin/python"
STATUS="$JOB_DIR/status.json"
STARTED_AT="$(date --iso-8601=seconds)"

write_status() {
  local state="$1"
  local detail="$2"
  local temporary="$STATUS.tmp"
  printf '{\n  "state": "%s",\n  "detail": "%s",\n  "pid": %s,\n  "pca_job_dir": "%s",\n  "spiral_job_dir": "%s",\n  "started_at": "%s",\n  "updated_at": "%s"\n}\n' \
    "$state" "$detail" "$$" "$PCA_JOB_DIR" "$SPIRAL_JOB_DIR" "$STARTED_AT" "$(date --iso-8601=seconds)" > "$temporary"
  mv "$temporary" "$STATUS"
}

on_exit() {
  local code=$?
  if [[ $code -ne 0 ]]; then
    write_status "failed" "exit_code_$code"
  fi
}
trap on_exit EXIT

write_status "waiting" "waiting_for_both_independent_runs"
while true; do
  PCA_STATE=$(sed -n 's/.*"state": "\([^"]*\)".*/\1/p' "$PCA_JOB_DIR/status.json" 2>/dev/null || true)
  SPIRAL_STATE=$(sed -n 's/.*"state": "\([^"]*\)".*/\1/p' "$SPIRAL_JOB_DIR/status.json" 2>/dev/null || true)
  if [[ "$PCA_STATE" == "failed" || "$SPIRAL_STATE" == "failed" ]]; then
    echo "one or both training jobs failed: pca=$PCA_STATE spiral=$SPIRAL_STATE" >&2
    exit 1
  fi
  if [[ "$PCA_STATE" == "complete" && "$SPIRAL_STATE" == "complete" ]]; then
    break
  fi
  sleep 30
done

for run_dir in "$PCA_RUN_DIR" "$SPIRAL_RUN_DIR"; do
  if [[ ! -f "$run_dir/evaluation/val/summary.json" || ! -f "$run_dir/evaluation/test/summary.json" ]]; then
    echo "run did not complete required validation/test outputs: $run_dir" >&2
    exit 1
  fi
done

write_status "comparing" "validation_and_test"
for split in val test; do
  "$PY" -u "$TASK/scripts/compare_results.py" \
    --split "$split" --allow-partial --evaluation-name evaluation \
    --run "$PCA_RUN_DIR" --run "$SPIRAL_RUN_DIR" \
    --output "$OUTPUT_ROOT/comparisons/pca_vs_spiral_direct_c4_s${SEED}_${split}.csv"
done
write_status "complete" "comparison_complete"
trap - EXIT
