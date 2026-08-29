#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: $0 BEST_CONFIG RUN_NAME [SEED]" >&2
  exit 2
fi

CONFIG=$1
RUN_NAME=$2
SEED=${3:-42}
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PYTHON_BIN=/home/jakaria/anaconda3/envs/pytorch_geo/bin/python

exec "$PYTHON_BIN" "$SCRIPT_DIR/train.py" \
  --config "$CONFIG" \
  --device cuda:1 \
  --seed "$SEED" \
  --run-name "$RUN_NAME"

