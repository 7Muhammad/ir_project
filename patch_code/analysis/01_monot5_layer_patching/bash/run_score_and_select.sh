#!/usr/bin/env bash
# bash/run_score_and_select.sh
# ============================
# Stage 01: score all pairs with monoT5 and select attacked examples.
#
# Usage:
#   bash bash/run_score_and_select.sh [config_path]

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

PAIRS_FILE="${OUTPUTS_DIR}/pairs/pairs.jsonl"
if [[ ! -f "${PAIRS_FILE}" ]]; then
    echo "ERROR: ${PAIRS_FILE} not found."
    echo "Please run Stage 00 first:  bash bash/run_prepare_pairs.sh ${CONFIG}"
    exit 1
fi

mkdir -p "${OUTPUTS_DIR}/scores"

python3 "${PROJECT_ROOT}/scripts/01_score_and_select.py" --config "${CONFIG}"

echo ""
echo "Outputs:"
echo "  ${OUTPUTS_DIR}/scores/all_scores.csv"
echo "  ${OUTPUTS_DIR}/scores/selected_examples.jsonl"
