#!/usr/bin/env bash
# LAMM (Tarasiou et al., CVPR 2024) on the ADNI left hippocampus: two Optuna studies over the
# same space, differing only in the backbone -- the paper's own two variants.
#
# The hypothesis being tested is not "transformers are good". It is that THE SPIRAL DECODER IS
# THE CEILING. LAMM's Table 1 shows SpiralNet++ losing to PCA on all four of its datasets,
# reproducing what we measure here; LAMM is the only method in that table that beats PCA, and
# what it changes is that both ends are transformer/MLPMixer with no convolution anywhere.
# Every one of the 19 MeshMAE runs in this project held the SpiralNet++ decoder fixed, so this
# component has never been varied.
#
# One study per GPU. They initially shared GPU 1, but at D=192/batch 16 that gave 4.3 s/epoch
# and trial 1 was cut off by the 1200s budget at epoch 282 of 400 -- while still UNDERFITTING,
# which is this model's actual failure mode (LAMM's first trials show gap 1.18-1.25x, matching
# PCA's 1.19x, with train error 0.031-0.066 against MeshMAE's 0.021). Truncating an
# underfitting model biases TPE toward small-D configs for finishing rather than for being
# better, so each study now gets a full card.
set -uo pipefail
PY=/home/jakaria/anaconda3/envs/pytorch_geo/bin/python
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN=/mnt/bulk10tb/Deep3DComp/ADNI_1_LHipp/task_lamm_ae_v1/run
mkdir -p "$RUN"
[[ -d /mnt/bulk10tb/Deep3DComp ]] || { echo "FATAL: /mnt/bulk10tb is not mounted"; exit 1; }

COMMON=(--epochs 400 --min-epochs 60 --patience 25 --trial-time-budget 1200 --latent 128)

launch () {  # study backbone gpu n_trials
  local S="$1" BB="$2" G="$3" N="${4:-20}"
  if [[ -f "$RUN/$S.pid" ]] && kill -0 "$(cat "$RUN/$S.pid")" 2>/dev/null; then
    echo "SKIP  $S (pid $(cat "$RUN/$S.pid"))"; return; fi
  setsid nohup "$PY" -u "$HERE/optuna_lamm.py" --study "$S" --backbone "$BB" \
      --gpu "$G" --n-trials "$N" "${COMMON[@]}" >>"$RUN/$S.out" 2>&1 </dev/null &
  sleep 3
  pgrep -f "optuna_lamm.py --study $S " | head -1 > "$RUN/$S.pid"
  echo "START $S backbone=$BB gpu=$G pid=$(cat "$RUN/$S.pid")"
}

# study            backbone      gpu  trials-to-add (SQLite studies resume, so this is a delta)
launch lamm_transformer transformer   1   20
launch lamm_mlpmixer    mlpmixer      0   18
echo; echo "logs: $RUN/lamm_*.out   csv: .../studies/lamm_*/trial_metrics.csv"
