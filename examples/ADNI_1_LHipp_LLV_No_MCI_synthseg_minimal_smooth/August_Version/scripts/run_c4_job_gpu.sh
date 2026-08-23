#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 6 ]]; then
  echo "usage: $0 CONFIG REPRESENTATION PHYSICAL_GPU SEED RUN_NAME JOB_DIR" >&2
  exit 2
fi

CONFIG="$1"
REPRESENTATION="$2"
PHYSICAL_GPU="$3"
SEED="$4"
RUN_NAME="$5"
JOB_DIR="$6"
TASK="examples/ADNI_1_LHipp_LLV_No_MCI_synthseg_minimal_smooth/August_Version"
PY="/home/jakaria/anaconda3/envs/pytorch_geo/bin/python"
OUTPUT_ROOT="/mnt/bulk10tb/Deep3DComp/ADNI_1_LHipp/August_Version"
RUN_DIR="$OUTPUT_ROOT/training/$REPRESENTATION/direct_c4/$RUN_NAME"
STATUS="$JOB_DIR/status.json"
STARTED_AT="$(date --iso-8601=seconds)"

write_status() {
  local state="$1"
  local detail="$2"
  local temporary="$STATUS.tmp"
  printf '{\n  "state": "%s",\n  "detail": "%s",\n  "pid": %s,\n  "physical_gpu": %s,\n  "representation": "%s",\n  "run_name": "%s",\n  "run_dir": "%s",\n  "started_at": "%s",\n  "updated_at": "%s"\n}\n' \
    "$state" "$detail" "$$" "$PHYSICAL_GPU" "$REPRESENTATION" "$RUN_NAME" "$RUN_DIR" "$STARTED_AT" "$(date --iso-8601=seconds)" > "$temporary"
  mv "$temporary" "$STATUS"
}

on_exit() {
  local code=$?
  if [[ $code -ne 0 ]]; then
    write_status "failed" "exit_code_$code"
  fi
}
trap on_exit EXIT

case "$REPRESENTATION" in
  pca128|spiralnet128|adaptive128) ;;
  *) echo "unsupported representation: $REPRESENTATION" >&2; exit 2 ;;
esac
if [[ ! -f "$CONFIG" ]]; then
  echo "missing config: $CONFIG" >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES="$PHYSICAL_GPU"
write_status "training" "direct_c4"
"$PY" -u "$TASK/scripts/train_c4.py" \
  --config "$CONFIG" --device cuda:0 --seed "$SEED" --run-name "$RUN_NAME"

write_status "evaluating_validation" "full_validation_bootstrap_2000"
"$PY" -u "$TASK/scripts/evaluate.py" \
  --run-dir "$RUN_DIR" --split val --device cuda:0 \
  --bootstrap-samples 2000 --evaluation-name evaluation

write_status "evaluating_test" "full_test_bootstrap_2000"
"$PY" -u "$TASK/scripts/evaluate.py" \
  --run-dir "$RUN_DIR" --split test --device cuda:0 \
  --bootstrap-samples 2000 --evaluation-name evaluation

write_status "complete" "training_validation_test_complete"
trap - EXIT

