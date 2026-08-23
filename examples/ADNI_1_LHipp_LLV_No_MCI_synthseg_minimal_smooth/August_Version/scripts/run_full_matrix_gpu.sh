#!/usr/bin/env bash
set -euo pipefail

# Long-running launcher for the user. This file is never invoked by smoke verification.
GPU="${1:-1}"
SEEDS="${2:-42 43 44}"
TASK="examples/ADNI_1_LHipp_LLV_No_MCI_synthseg_minimal_smooth/August_Version"

for SEED in $SEEDS; do
  for REP in pca128 spiralnet128 adaptive128; do
    for METHOD in plain_ode brainode direct_c4; do
      CONFIG="$TASK/configs/${REP}_${METHOD}_s42.json"
      RUN_NAME="${REP}_${METHOD}_s${SEED}"
      "$TASK/scripts/run_one_gpu.sh" "$CONFIG" "$GPU" "$SEED" "$RUN_NAME"
    done
  done
done
