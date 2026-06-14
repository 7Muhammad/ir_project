#!/usr/bin/env bash
# bash/run_prepare_pairs.sh
# ==========================
# Stage 00: prepare the normalised query-document pairs JSONL file.
#
# Usage:
#   bash bash/run_prepare_pairs.sh [config_path]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

CONFIG="${1:-${PROJECT_ROOT}/configs/default.yaml}"

# Read outputs.base_dir from the config so this script works with any config
# (e.g. configs/smoke.yaml writes to outputs_smoke/ instead of outputs/).
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

echo "============================================================"
echo "  Stage 00: prepare_pairs"
echo "  Config: ${CONFIG}"
echo "============================================================"

mkdir -p "${OUTPUTS_DIR}/pairs"
mkdir -p "${OUTPUTS_DIR}/data_discovery"

python3 "${PROJECT_ROOT}/scripts/00_prepare_pairs.py" --config "${CONFIG}"

echo ""
echo "Output: ${OUTPUTS_DIR}/pairs/pairs.jsonl"
