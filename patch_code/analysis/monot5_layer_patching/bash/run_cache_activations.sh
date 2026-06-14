#!/usr/bin/env bash
# bash/run_cache_activations.sh
# ==============================
# Stage 02: run control and attack forward passes and cache layer activations.
#
# Usage:
#   bash bash/run_cache_activations.sh [config_path]

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
if [[ ! -f "${SELECTED_FILE}" ]]; then
    echo "ERROR: ${SELECTED_FILE} not found."
    echo "Please run Stage 01 first:  bash bash/run_score_and_select.sh ${CONFIG}"
    exit 1
fi

mkdir -p "${OUTPUTS_DIR}/activations"

python3 "${PROJECT_ROOT}/scripts/02_cache_activations.py" --config "${CONFIG}"

echo ""
echo "Output: ${OUTPUTS_DIR}/activations/"
