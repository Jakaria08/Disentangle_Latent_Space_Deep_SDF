#!/usr/bin/env bash
# Experiments 1 and 2, as two Optuna studies over the SAME hyperparameter space.
#
#   search_flatten   head = flatten  (1.409M dense token->latent path)
#   search_grouped   head = grouped  (weight-shared per-patch projection, rank searched)
#
# Running both as searches -- rather than comparing two untuned single runs -- means the
# flatten-vs-grouped question is answered after each has had equal tuning. Closing that
# asymmetry is the whole point: spiral got 138 trials, MeshMAE got 0.
#
# Resume: re-run this script; each SQLite study continues where it stopped.
set -uo pipefail
PY=/home/jakaria/anaconda3/envs/pytorch_geo/bin/python
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN=/mnt/bulk10tb/Deep3DComp/ADNI_1_LHipp/task_meshmae_ae_v1/run
mkdir -p "$RUN"
[[ -d /mnt/bulk10tb/Deep3DComp ]] || { echo "FATAL: /mnt/bulk10tb is not mounted"; exit 1; }

# 220 epochs: every EMA run so far converged and early-stopped by ~240 (S9 248, S8 240,
# S8c 240, E2 198), and the EMA curve is flat well before that (S9 was 0.0387 at epoch 150
# vs 0.038463 at 248). 1100s clears the slowest config in the space -- d_model 128, depth 6,
# batch 16, i.e. exactly S9 at 4.3 s/epoch -- so NO trial is scored before it converges.
COMMON=(--n-trials 20 --epochs 220 --min-epochs 80 --patience 15 --trial-time-budget 1100)

launch () {  # study head gpu
  local S="$1" H="$2" G="$3"
  if [[ -f "$RUN/$S.pid" ]] && kill -0 "$(cat "$RUN/$S.pid")" 2>/dev/null; then
    echo "SKIP  $S (pid $(cat "$RUN/$S.pid"))"; return; fi
  setsid nohup "$PY" -u "$HERE/optuna_meshmae.py" --study "$S" --head "$H" --gpu "$G" \
      "${COMMON[@]}" >"$RUN/$S.out" 2>&1 </dev/null &
  sleep 3
  pgrep -f "optuna_meshmae.py --study $S " | head -1 > "$RUN/$S.pid"
  echo "START $S head=$H gpu=$G pid=$(cat "$RUN/$S.pid")"
}

launch search_flatten flatten 0
launch search_grouped grouped 2
echo; echo "logs: $RUN/search_*.out   csv: .../studies/search_*/trial_metrics.csv"
