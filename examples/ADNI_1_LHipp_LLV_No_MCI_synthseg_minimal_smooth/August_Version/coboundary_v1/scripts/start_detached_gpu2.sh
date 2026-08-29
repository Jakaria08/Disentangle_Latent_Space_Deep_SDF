#!/usr/bin/env bash
set -euo pipefail

TASK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BULK_ROOT="/mnt/bulk10tb/Deep3DComp/ADNI_1_LHipp/August_Version"
SESSION_NAME="exact_cob_s42_gpu2"
SCREEN_LOG="${BULK_ROOT}/exact_coboundary_gpu2_screen.log"

if screen -ls 2>/dev/null | grep -Fq ".${SESSION_NAME}"; then
  echo "Refusing duplicate active screen session: ${SESSION_NAME}" >&2
  exit 2
fi

screen -L -Logfile "${SCREEN_LOG}" -dmS "${SESSION_NAME}" \
  bash "${TASK_ROOT}/scripts/launch_gpu2.sh"

echo "session=${SESSION_NAME}"
echo "screen_log=${SCREEN_LOG}"
echo "status: screen -ls"
echo "follow: tail -f ${SCREEN_LOG}"

