#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: bash $0 SCRIPT.py [arguments ...]" >&2
    exit 2
fi

temp_root="${EXACT_SDF_BULK_TMPDIR:-/mnt/bulk10tb/.xsdf_tmp}"
cache_root="${EXACT_SDF_BULK_CACHEDIR:-/mnt/bulk10tb/.xsdf_cache}"

for path in "${temp_root}" "${cache_root}"; do
    case "${path}" in
        /mnt/bulk10tb/*) ;;
        *)
            echo "Refusing non-bulk runtime path: ${path}" >&2
            exit 2
            ;;
    esac
done

# Python multiprocessing appends `pymp-*/listener-*` to TMPDIR. Linux limits
# AF_UNIX socket paths to roughly 108 bytes, so retain ample suffix headroom.
if (( ${#temp_root} > 60 )); then
    echo "TMPDIR is too long for Python multiprocessing sockets: ${temp_root}" >&2
    exit 2
fi

mkdir -p \
    "${temp_root}" \
    "${cache_root}/cuda" \
    "${cache_root}/matplotlib" \
    "${cache_root}/numba" \
    "${cache_root}/torch" \
    "${cache_root}/triton" \
    "${cache_root}/xdg"

export PYTHONDONTWRITEBYTECODE=1
export TMPDIR="${temp_root}"
export TEMP="${temp_root}"
export TMP="${temp_root}"
export JOBLIB_TEMP_FOLDER="${temp_root}"
export XDG_CACHE_HOME="${cache_root}/xdg"
export TORCH_HOME="${cache_root}/torch"
export MPLCONFIGDIR="${cache_root}/matplotlib"
export CUDA_CACHE_PATH="${cache_root}/cuda"
export NUMBA_CACHE_DIR="${cache_root}/numba"
export TRITON_CACHE_DIR="${cache_root}/triton"

exec /home/jakaria/anaconda3/envs/inr_sdf/bin/python "$@"
