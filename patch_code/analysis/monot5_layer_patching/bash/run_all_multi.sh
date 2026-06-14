#!/usr/bin/env bash
# bash/run_all_multi.sh
# ======================
# Run the full multi-attack experiment end-to-end:
#   1. Multi-attack pipeline (stages 00–04 for each attack)
#   2. Cross-attack comparison and plots
#
# Usage:
#   conda activate advseq2seq
#   bash bash/run_all_multi.sh [config_path]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

CONFIG="${1:-${PROJECT_ROOT}/configs/default.yaml}"

if [[ ! -f "${CONFIG}" ]]; then
    echo "ERROR: Config file not found: ${CONFIG}"
    echo "Usage: bash bash/run_all_multi.sh [path/to/config.yaml]"
    exit 1
fi

echo "============================================================"
echo "  monoT5 Multi-Attack Experiment (full run)"
echo "  Config: ${CONFIG}"
echo "============================================================"
echo ""

echo "--- Phase 1: Multi-attack pipeline ---"
bash "${SCRIPT_DIR}/run_multi_attack_pipeline.sh" "${CONFIG}"

echo ""
echo "--- Phase 2: Cross-attack comparison ---"
bash "${SCRIPT_DIR}/run_compare_attacks.sh" "${CONFIG}"

echo ""
echo "============================================================"
echo "  Done.  Check outputs/attacks/ and outputs/attack_comparison/"
echo "============================================================"
