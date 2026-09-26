#!/usr/bin/env bash
# bash/run_layer_patching.sh
# ===========================
# Stage 03: run layer-level activation patching.
#
# Usage:
#   bash bash/run_layer_patching.sh [config_path]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

CONFIG="${1:-${PROJECT_ROOT}/configs/default.yaml}"

# Read outputs.base_dir from the config so this script works with any config.
OUTPUTS_DIR="$(python3 -c "
import yaml, pathlib
with open('${CONFIG}') as f:
    cfg = yaml.safe_load(f)
base = cfg.get('outputs', {}).get('base_dir', 'outputs')
p = pathlib.Path(base)
if not p.is_absolute():
    p = pathlib.Path('${PROJECT_ROOT}') / p
print(str(p))
")"
echo "  Config: ${CONFIG}"
echo "============================================================"

SELECTED_FILE="${OUTPUTS_DIR}/scores/selected_examples.jsonl"
ACT_DIR="${OUTPUTS_DIR}/activations"

if [[ ! -f "${SELECTED_FILE}" ]]; then
    echo "ERROR: ${SELECTED_FILE} not found."
    echo "Please run Stage 01 first:  bash bash/run_score_and_select.sh ${CONFIG}"
    exit 1
fi

if [[ ! -d "${ACT_DIR}" ]] || [[ -z "$(ls -A "${ACT_DIR}" 2>/dev/null)" ]]; then
    echo "ERROR: No activation files found in ${ACT_DIR}."
    echo "Please run Stage 02 first:  bash bash/run_cache_activations.sh ${CONFIG}"
    exit 1
fi

mkdir -p "${OUTPUTS_DIR}/patching"

python3 "${PROJECT_ROOT}/scripts/03_run_layer_patching.py" --config "${CONFIG}"

echo ""
echo "Outputs:"
echo "  ${OUTPUTS_DIR}/patching/layer_patching_results_detailed.csv"
echo "  ${OUTPUTS_DIR}/patching/layer_patching_results.csv"
