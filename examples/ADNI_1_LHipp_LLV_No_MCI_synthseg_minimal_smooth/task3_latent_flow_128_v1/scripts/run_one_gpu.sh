#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 4 ]]; then
  echo "usage: $0 CONFIG GPU [SEED] [RUN_NAME]" >&2
  exit 2
fi

CONFIG="$1"
GPU="$2"
SEED="${3:-42}"
RUN_NAME="${4:-}"
TASK="examples/ADNI_1_LHipp_LLV_No_MCI_synthseg_minimal_smooth/task3_latent_flow_128_v1"
PY="/home/jakaria/anaconda3/envs/pytorch_geo/bin/python"

case "$(basename "$CONFIG")" in
  *direct_c4*) TRAINER="$TASK/scripts/train_c4.py" ;;
  *plain_ode*|*brainode*) TRAINER="$TASK/scripts/train_latent_ode.py" ;;
  *) echo "cannot infer trainer from config name: $CONFIG" >&2; exit 2 ;;
esac

ARGS=(--config "$CONFIG" --device cuda:0 --seed "$SEED")
if [[ -n "$RUN_NAME" ]]; then ARGS+=(--run-name "$RUN_NAME"); fi
CUDA_VISIBLE_DEVICES="$GPU" "$PY" "$TRAINER" "${ARGS[@]}"
