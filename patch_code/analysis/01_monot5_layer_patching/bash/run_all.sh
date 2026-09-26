#!/usr/bin/env bash
# bash/run_all.sh
# ================
# Run the complete activation-patching experiment end-to-end.
#
# Usage:
#   conda activate advseq2seq
#   bash bash/run_all.sh [config_path]
#
# Stages:
#   find_data  →  00_prepare_pairs  →  01_score_and_select
#               →  02_cache_activations  →  03_run_layer_patching  →  04_plot_results
#
# Before running:
#   1. Activate the conda environment:
#        conda activate advseq2seq
#   2. Optionally edit configs/default.yaml (upstream_repo_path is pre-configured).
#   3. Optionally run bash bash/find_data.sh to verify data discovery.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

CONFIG="${1:-${PROJECT_ROOT}/configs/default.yaml}"

if [[ ! -f "${CONFIG}" ]]; then
    echo "ERROR: Config file not found: ${CONFIG}"
    echo "Usage: bash bash/run_all.sh [path/to/config.yaml]"
    exit 1
fi

echo "============================================================"
echo "  monoT5 Activation Patching Experiment"
echo "  Config: ${CONFIG}"
echo "============================================================"
echo ""

echo "--- Stage: find_data ---"
bash "${SCRIPT_DIR}/find_data.sh" "${CONFIG}"

echo ""
echo "--- Stage 00: prepare_pairs ---"
bash "${SCRIPT_DIR}/run_prepare_pairs.sh" "${CONFIG}"

echo ""
echo "--- Stage 01: score_and_select ---"
bash "${SCRIPT_DIR}/run_score_and_select.sh" "${CONFIG}"

echo ""
echo "--- Stage 02: cache_activations ---"
bash "${SCRIPT_DIR}/run_cache_activations.sh" "${CONFIG}"

echo ""
echo "--- Stage 03: layer_patching ---"
bash "${SCRIPT_DIR}/run_layer_patching.sh" "${CONFIG}"

echo ""
echo "--- Stage 04: plot_results ---"
bash "${SCRIPT_DIR}/run_plots.sh" "${CONFIG}"

echo ""
echo "============================================================"
echo "  Experiment complete!"
echo "  Results in: ${PROJECT_ROOT}/outputs/"
echo "============================================================"
