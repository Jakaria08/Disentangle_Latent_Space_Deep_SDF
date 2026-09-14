#!/usr/bin/env bash
# Stage 3 finisher: wait for the ADNI training suite, reconcile it, then run the analysis suite.
#
#   setsid nohup bash scripts/run_stage3_analysis_after_training.sh > /dev/null 2>&1 < /dev/null &
#
# 1. Waits until adni_training has no pending or running jobs.
# 2. Refuses to continue if any job that is not "ok" lacks its done marker (a real, unrepaired failure).
#    Jobs repaired by hand (e.g. an evaluation rerun after a GPU out-of-memory) have their marker.
# 3. Waits for the training orchestrator process to exit, then reruns it once so repaired jobs are
#    recorded as ok from their markers (nothing that already finished is rerun).
# 4. Builds the analysis job file (condition sweeps for all 60 seed-runs + report) and runs it.
# Every step is logged to stage3_adni/jobs/finisher.log. GPUs 0 and 2 only.
set -uo pipefail

TASK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
S3="/mnt/bulk10tb/Deep3DComp/LHipp_LatentDynamics_CrossCohort_v1/stage3_adni"
PY="/home/jakaria/anaconda3/envs/pytorch_geo/bin/python"
TRAIN_JOBS="$S3/jobs/adni_training.json"
LOG="$S3/jobs/finisher.log"
cd "$TASK_ROOT/scripts"
say() { echo "$(date '+%F %T') $*" >> "$LOG"; }

say "finisher started (pid $$)"
until "$PY" -c "
import json, sys
c = json.load(open('${TRAIN_JOBS%.json}.state.json'))['counts']
sys.exit(0 if not (c.get('pending') or c.get('running')) else 1)" 2>/dev/null; do
  sleep 60
done
say "training suite has no pending or running jobs"

if ! "$PY" - "$TRAIN_JOBS" >> "$LOG" 2>&1 <<'PY'
import json, sys
from pathlib import Path
jobs_path = Path(sys.argv[1])
jobs = {j["id"]: j for j in json.load(open(jobs_path))["jobs"]}
state = json.load(open(jobs_path.with_suffix(".state.json")))["jobs"]
def done(marker):
    if isinstance(marker, str):
        return Path(marker).exists()
    path = Path(marker["path"])
    return path.is_file() and json.load(open(path)).get(marker["key"]) == marker["equals"]
unrepaired = [k for k, v in state.items() if v["status"] != "ok" and not done(jobs[k].get("done_marker"))]
print("non-ok jobs:", [k for k, v in state.items() if v["status"] != "ok"], "| unrepaired:", unrepaired)
sys.exit(1 if unrepaired else 0)
PY
then
  say "STOP: a training-suite job failed and has no output; analysis not started"
  exit 1
fi

while ps -eo args | grep -q "[o]rchestrate_dynamics.py --jobs $TRAIN_JOBS"; do
  sleep 20
done
say "training orchestrator exited; reconciling repaired jobs"
"$PY" orchestrate_dynamics.py --jobs "$TRAIN_JOBS" --gpus 0,2 >> "$LOG" 2>&1
say "reconcile exit code $?"

"$PY" stage3_build_jobs.py analysis >> "$LOG" 2>&1 || { say "STOP: could not build analysis jobs"; exit 1; }
say "analysis suite starting"
"$PY" orchestrate_dynamics.py --jobs "$S3/jobs/adni_analysis.json" --gpus 0,2 --capacity-per-gpu 3 >> "$LOG" 2>&1
code=$?
say "analysis suite finished with exit code $code"
[ -f "$S3/reports/stage3_adni_report.md" ] && say "report: $S3/reports/stage3_adni_report.md"
exit $code
