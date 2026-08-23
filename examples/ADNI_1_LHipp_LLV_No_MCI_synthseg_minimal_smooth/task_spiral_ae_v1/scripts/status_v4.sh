#!/usr/bin/env bash
# v4 status. Usage: ./status_v4.sh [TAG]
set -uo pipefail
TAG="${1:-v4}"
B=/mnt/bulk10tb/Deep3DComp/ADNI_1_LHipp/task_spiral_ae_v1
PY=/home/jakaria/anaconda3/envs/pytorch_geo/bin/python

echo "PCA reference (test):  K=128 -> 0.034380    K=256 -> 0.010787"
echo
for L in A:0:"spiralnet_z128 adaptive_z128 mlp_z128" B:2:"spiralnet_z256 adaptive_z256 mlp_z256"; do
  IFS=: read -r LANE GPU EXPS <<<"$L"
  P="$B/run/lane_${LANE}.pid"
  if [[ -f "$P" ]] && kill -0 "$(cat "$P")" 2>/dev/null; then S="running"; else S="stopped"; fi
  echo "lane $LANE (gpu $GPU) [$S]"
  for E in $EXPS; do
    K="${E##*_z}"; case "$K" in 128) R=0.034380;; *) R=0.010787;; esac
    if [[ -f "$B/run/${E}_${TAG}.done" ]]; then ST="done"; else
      [[ -d "$B/studies/${E}_${TAG}" ]] && ST="running" || ST="queued"; fi
    N="-"; BEST="-"; VERD=""
    CSV="$B/studies/${E}_${TAG}/trial_metrics.csv"
    [[ -f "$CSV" ]] && read -r N BEST < <("$PY" - "$CSV" <<'EOF'
import csv,sys
r=[x for x in csv.DictReader(open(sys.argv[1])) if x.get("val_rmse_mm")]
v=sorted(float(x["val_rmse_mm"]) for x in r)
print(len(r), f"{v[0]:.6f}" if v else "-")
EOF
)
    J="$B/best/${E}_${TAG}/best_summary.json"
    [[ -f "$J" ]] && VERD=$("$PY" -c "import json;d=json.load(open('$J'));print(' | k=%s test=%.6f+-%.6f %s'%(d['k'],d['seed_repeats']['test_mean'],d['seed_repeats']['test_sd'],d['verdict']))")
    printf "   %-16s %-8s trials=%-4s best_val=%-10s%s\n" "$E" "$ST" "$N" "$BEST" "$VERD"
  done
done
echo
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
