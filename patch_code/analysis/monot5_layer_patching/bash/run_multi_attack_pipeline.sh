#!/usr/bin/env bash
# bash/run_multi_attack_pipeline.sh
# ===================================
# Run the multi-attack activation-patching pipeline.
# Discovers attacks from the config's 'attacks' block, loads the model once,
# and runs stages 00–04 for each attack, writing outputs to
#   outputs/attacks/{attack_name}/
#
# Usage:
#   conda activate advseq2seq
#   bash bash/run_multi_attack_pipeline.sh [config_path]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

CONFIG="${1:-${PROJECT_ROOT}/configs/default.yaml}"

if [[ ! -f "${CONFIG}" ]]; then
    echo "ERROR: Config file not found: ${CONFIG}"
    echo "Usage: bash bash/run_multi_attack_pipeline.sh [path/to/config.yaml]"
    exit 1
fi

echo "============================================================"
echo "  monoT5 Multi-Attack Activation Patching"
echo "  Config: ${CONFIG}"
echo "============================================================"
echo ""

python3 "${PROJECT_ROOT}/scripts/10_run_multi_attack_pipeline.py" --config "${CONFIG}"
