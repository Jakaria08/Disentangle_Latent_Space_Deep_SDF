#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 5 ]]; then
  echo "Usage: $0 {spiral|adaptive} [study_tag] [trials] [trial_epochs] [samples_per_epoch]" >&2
  exit 2
fi

OPERATOR=$1
STUDY_TAG=${2:-main_v2}
TRIALS=${3:-16}
TRIAL_EPOCHS=${4:-80}
SAMPLES_PER_EPOCH=${5:-384}
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PYTHON_BIN=/home/jakaria/anaconda3/envs/pytorch_geo/bin/python

exec "$PYTHON_BIN" "$SCRIPT_DIR/optuna_search_v2.py" \
  --operator "$OPERATOR" \
  --device cuda:1 \
  --study-tag "$STUDY_TAG" \
  --n-trials "$TRIALS" \
  --trial-epochs "$TRIAL_EPOCHS" \
  --trial-samples-per-epoch "$SAMPLES_PER_EPOCH"
