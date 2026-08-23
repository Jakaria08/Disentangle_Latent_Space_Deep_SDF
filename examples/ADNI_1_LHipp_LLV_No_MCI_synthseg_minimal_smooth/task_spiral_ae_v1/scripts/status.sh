#!/usr/bin/env bash
# Status of the four studies. Usage: ./status.sh [TAG]   (default TAG=v2)
set -uo pipefail

TAG="${1:-v2}"
B=/mnt/bulk10tb/Deep3DComp/ADNI_1_LHipp/task_spiral_ae_v1
PY=/home/jakaria/anaconda3/envs/pytorch_geo/bin/python

echo "PCA targets (val):  z128 = 0.033668   z256 = 0.010338"
echo
printf "%-22s %-8s %-7s %-11s %-11s %s\n" experiment state trials best_val vs_PCA current
for EXP in spiralnet_z128 spiralnet_z256 adaptive_z128 adaptive_z256; do
  KEY="${EXP}_${TAG}"
  PIDFILE="$B/run/${KEY}.pid"
  if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then STATE=running; else STATE=stopped; fi
  CSV="$B/studies/$KEY/trial_metrics.csv"
  if [[ -f "$CSV" ]]; then
    read -r N BEST < <("$PY" - "$CSV" <<'EOF'
import csv,sys
rows=[r for r in csv.DictReader(open(sys.argv[1])) if r.get("val_rmse_mm")]
vals=sorted(float(r["val_rmse_mm"]) for r in rows)
print(len(rows), f"{vals[0]:.6f}" if vals else "-")
EOF
)
  else N=0; BEST="-"; fi
  case "$EXP" in *z128) P=0.033668;; *) P=0.010338;; esac
  if [[ "$BEST" != "-" ]]; then RATIO=$(awk -v b="$BEST" -v p="$P" 'BEGIN{printf "%.2fx", b/p}'); else RATIO="-"; fi
  CUR=$(tail -1 "$B/run/${KEY}.out" 2>/dev/null | grep -o "best=[0-9.]*" | head -1)
  printf "%-22s %-8s %-7s %-11s %-11s %s\n" "$KEY" "$STATE" "$N" "$BEST" "$RATIO" "${CUR:--}"
done
echo
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
