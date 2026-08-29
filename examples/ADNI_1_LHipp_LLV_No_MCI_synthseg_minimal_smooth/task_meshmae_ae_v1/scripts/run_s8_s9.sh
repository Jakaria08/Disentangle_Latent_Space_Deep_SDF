#!/usr/bin/env bash
# S8 / S9: close the last 5.4% of gap needed to beat SpiralNet++ (val 0.036784).
#
# From S7 (train 0.017318, val 0.038889, gap 2.245x) the arithmetic is fixed: since
# val = train x gap, beating SpiralNet++ needs gap <= 2.124x. S5 proved a regulariser
# cannot deliver that -- it cut the gap 28% but raised train error 44%, a net loss. Only a
# better prior or better weights move val without paying that toll.
#
#   S9_ema        EMA of the weights. S7's last 100 epochs are flat to five decimals, so
#                 the optimiser is oscillating in a basin; averaging returns its centre.
#                 Also re-runs S7's exact config, so its live curve must reproduce 0.038889.
#   S8_distbias   Learned locality bias -softplus(w)*d(i,j) on attention. The measured
#                 deficit vs SpiralNet++ IS locality (gap 1.41x vs 2.245x). Previously
#                 tested ONLY in S3, on top of the latentset head that scored 0.1159 -- a
#                 5% effect is invisible through a 175% failure, so this is untested, not
#                 retested.
#   S8c_biasoff   Control, queued behind S9. --distance-bias also swaps nn.TransformerEncoder
#                 for BiasedEncoderBlock; bias_init=-10 gives softplus~4.5e-5, i.e. the
#                 custom block with the bias off. S8 - S8c is the locality effect, clean.
set -uo pipefail
PY=/home/jakaria/anaconda3/envs/pytorch_geo/bin/python
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN=/mnt/bulk10tb/Deep3DComp/ADNI_1_LHipp/task_meshmae_ae_v1/run
mkdir -p "$RUN"
[[ -d /mnt/bulk10tb/Deep3DComp ]] || { echo "FATAL: /mnt/bulk10tb is not mounted"; exit 1; }

COMMON=(--patch-level 3 --head flatten --tokenizer both --noise-std 0.0
        --no-freeze-decoder --warm-start-decoder --ema-decay 0.999
        --epochs 500 --eval-every 1 --min-epochs 150 --patience 25
        --lr 3e-4 --weight-decay 0.05 --batch-size 16)

launch () {  # name gpu extra...
  local NAME="$1" GPU="$2"; shift 2
  setsid nohup "$PY" -u "$HERE/train_meshmae.py" --run-name "$NAME" --gpu "$GPU" \
      "${COMMON[@]}" "$@" >"$RUN/$NAME.out" 2>&1 </dev/null &
  sleep 3                                   # setsid forks; find the real python pid
  pgrep -f "train_meshmae.py --run-name $NAME " > "$RUN/$NAME.pid"
  echo "START $NAME gpu=$GPU pid=$(cat "$RUN/$NAME.pid" | tr '\n' ' ') $*"
}

launch S9_ema      0
launch S8_distbias 2 --distance-bias --bias-init 1.0

# control follows S9 on the same card, detached so it survives this shell
setsid nohup bash -c "
  while kill -0 \$(cat '$RUN/S9_ema.pid') 2>/dev/null; do sleep 30; done
  '$PY' -u '$HERE/train_meshmae.py' --run-name S8c_biasoff --gpu 0 ${COMMON[*]} \
      --distance-bias --bias-init -10 > '$RUN/S8c_biasoff.out' 2>&1
" >"$RUN/queue_s8c.out" 2>&1 </dev/null &
disown || true
echo "QUEUE S8c_biasoff gpu=0 (starts when S9_ema exits)"
