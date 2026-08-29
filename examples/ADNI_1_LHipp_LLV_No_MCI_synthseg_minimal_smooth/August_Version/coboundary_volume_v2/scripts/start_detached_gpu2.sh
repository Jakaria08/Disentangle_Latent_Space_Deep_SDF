#!/usr/bin/env bash
set -euo pipefail

TASK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BULK_ROOT="/mnt/bulk10tb/Deep3DComp/ADNI_1_LHipp/August_Version"
SESSION="volume_cob_v2_s42_gpu2"
STAMP="$(date +%Y%m%d_%H%M%S)"
SCREEN_LOG="${BULK_ROOT}/launches/${SESSION}_screen_${STAMP}.log"

if screen -ls 2>/dev/null | grep -q "[.]${SESSION}[[:space:]]"; then
  echo "Session already exists: ${SESSION}" >&2
  exit 2
fi
mkdir -p "${BULK_ROOT}/launches"
screen -dmS "${SESSION}" -L -Logfile "${SCREEN_LOG}" bash -lc "exec '${TASK_ROOT}/scripts/launch_gpu2.sh'"
sleep 2
if ! screen -ls 2>/dev/null | grep -q "[.]${SESSION}[[:space:]]"; then
  echo "Detached launcher exited early; inspect ${SCREEN_LOG}" >&2
  exit 3
fi
echo "started session=${SESSION} screen_log=${SCREEN_LOG}"
