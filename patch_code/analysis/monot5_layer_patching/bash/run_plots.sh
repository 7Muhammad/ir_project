#!/usr/bin/env bash
# bash/run_plots.sh
# ==================
# Stage 04: generate all plots from patching results.
#
# Usage:
#   bash bash/run_plots.sh [config_path]

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

AGG_FILE="${OUTPUTS_DIR}/patching/layer_patching_results.csv"
if [[ ! -f "${AGG_FILE}" ]]; then
    echo "ERROR: ${AGG_FILE} not found."
    echo "Please run Stage 03 first:  bash bash/run_layer_patching.sh ${CONFIG}"
    exit 1
fi

mkdir -p "${OUTPUTS_DIR}/plots"

python3 "${PROJECT_ROOT}/scripts/04_plot_results.py" --config "${CONFIG}"

echo ""
echo "Outputs:"
echo "  ${OUTPUTS_DIR}/plots/attack_delta_hist.png"
echo "  ${OUTPUTS_DIR}/plots/forward_layer_component_heatmap.png"
echo "  ${OUTPUTS_DIR}/plots/reverse_layer_component_heatmap.png"
echo "  ${OUTPUTS_DIR}/plots/combined_layer_component_heatmap.png"
