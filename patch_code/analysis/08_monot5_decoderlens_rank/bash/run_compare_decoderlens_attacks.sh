#!/usr/bin/env bash
# bash/run_compare_decoderlens_attacks.sh
# =======================================
# Re-run ONLY the cross-attack comparison (Stage 06) over already-computed
# per-attack results.  Useful after tweaking comparison plots.
#
# Usage:
#   conda activate advseq2seq
#   bash bash/run_compare_decoderlens_attacks.sh [config_path]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

CONFIG="${1:-${PROJECT_ROOT}/configs/default.yaml}"

if [[ ! -f "${CONFIG}" ]]; then
    echo "ERROR: Config file not found: ${CONFIG}"
    exit 1
fi

echo "--- Stage 06: cross-attack comparison ---"
python "${PROJECT_ROOT}/scripts/06_compare_decoderlens_attacks.py" --config "${CONFIG}"
