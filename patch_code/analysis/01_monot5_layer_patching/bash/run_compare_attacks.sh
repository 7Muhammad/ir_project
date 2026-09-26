#!/usr/bin/env bash
# bash/run_compare_attacks.sh
# ============================
# Collect results from all successful attack runs and produce cross-attack
# comparison summaries and plots in outputs/attack_comparison/.
#
# Usage:
#   conda activate advseq2seq
#   bash bash/run_compare_attacks.sh [config_path]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

CONFIG="${1:-${PROJECT_ROOT}/configs/default.yaml}"

if [[ ! -f "${CONFIG}" ]]; then
    echo "ERROR: Config file not found: ${CONFIG}"
    echo "Usage: bash bash/run_compare_attacks.sh [path/to/config.yaml]"
    exit 1
fi

echo "============================================================"
echo "  monoT5 Cross-Attack Comparison"
echo "  Config: ${CONFIG}"
echo "============================================================"
echo ""

python3 "${PROJECT_ROOT}/scripts/11_compare_attacks.py" --config "${CONFIG}"
