#!/usr/bin/env bash
# Reuse the repository's canonical SDF preprocessor so it finds bin/PreprocessMesh.
set -euo pipefail

SDF_REPO_DIR="$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
exec python "$SDF_REPO_DIR/preprocess_data.py" "$@"
