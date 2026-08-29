#!/usr/bin/env bash
# S5 / S6: close the MeshMAE generalisation gap.
#
# S2_flatten ended at train 0.017323 / val 0.042143 -- a 2.43x gap, against PCA-128's 1.19x.
# Its TRAIN error already beats PCA-128's train error (0.028247), so fitting power is surplus
# and the whole remaining deficit is generalisation. Both runs attack that, and both hold
# every S2 hyper-parameter fixed so each is a single-variable change.
#
#   S5  input-noise augmentation. Every top v7 trial chose noise_std 0.06-0.08; S2 used 0.0.
#   S6  raw patch tokenizer + the same noise. S2's moments compress 8238 coordinates to
#       86 x 12 = 1032 before the transformer sees them; "both" makes the input a strict
#       superset of S2's, so a regression can only be optimisation, not lost signal.
#
# Stop:  kill $(cat /mnt/bulk10tb/.../task_meshmae_ae_v1/run/S{5,6}_*.pid)
set -euo pipefail
PY=/home/jakaria/anaconda3/envs/pytorch_geo/bin/python
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN=/mnt/bulk10tb/Deep3DComp/ADNI_1_LHipp/task_meshmae_ae_v1/run
mkdir -p "$RUN"
[[ -d /mnt/bulk10tb/Deep3DComp ]] || { echo "FATAL: /mnt/bulk10tb is not mounted"; exit 1; }

COMMON=(--patch-level 3 --head flatten --no-freeze-decoder --warm-start-decoder
        --epochs 500 --eval-every 1 --min-epochs 150 --patience 25
        --lr 3e-4 --weight-decay 0.05 --batch-size 16 --noise-std 0.06)

launch () {  # name gpu extra...
  local NAME="$1" GPU="$2"; shift 2
  if [[ -f "$RUN/$NAME.pid" ]] && kill -0 "$(cat "$RUN/$NAME.pid")" 2>/dev/null; then
    echo "SKIP  $NAME (pid $(cat "$RUN/$NAME.pid"))"; return; fi
  setsid nohup "$PY" -u "$HERE/train_meshmae.py" --run-name "$NAME" --gpu "$GPU" \
      "${COMMON[@]}" "$@" >"$RUN/$NAME.out" 2>&1 </dev/null &
  echo $! >"$RUN/$NAME.pid"; disown || true
  echo "START $NAME gpu=$GPU pid=$(cat "$RUN/$NAME.pid")  $*"
}

launch S5_noise06     0 --tokenizer moments
launch S6_raw_noise06 2 --tokenizer both
echo; echo "logs: $RUN/S5_noise06.out  $RUN/S6_raw_noise06.out"
