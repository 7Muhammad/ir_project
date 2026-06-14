#!/usr/bin/env bash
# bash/run_smoke_test.sh
# ======================
# Run the full activation-patching pipeline on a tiny number of examples.
#
# PURPOSE
# -------
# A smoke test checks that the code runs end-to-end without crashing.
# It uses configs/smoke.yaml which limits the experiment to 5 pairs and
# 2 selected examples — fast enough to run in a few minutes on CPU.
#
# If this script completes successfully, all five pipeline stages work.
# The results will not be scientifically meaningful (too few examples),
# but errors will surface before you commit hours of GPU time.
#
# Usage:
#   conda activate advseq2seq
#   bash bash/run_smoke_test.sh [config_path]
#
# Default config: configs/smoke.yaml
# Output directory: outputs_smoke/  (set in configs/smoke.yaml)
#
# At the end, this script runs the sanity-check script to print diagnostics.
# It also verifies that all expected output files exist.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

CONFIG="${1:-${PROJECT_ROOT}/configs/smoke.yaml}"

if [[ ! -f "${CONFIG}" ]]; then
    echo "ERROR: Config file not found: ${CONFIG}"
    echo "Usage: bash bash/run_smoke_test.sh [path/to/config.yaml]"
    exit 1
fi

# ---- Read base_dir from config (Python is already required for the pipeline) --
# We use Python to parse the YAML so we don't need to install a bash YAML parser.
OUTPUTS_DIR="$(python3 - <<EOF
import sys, yaml
with open("${CONFIG}") as f:
    cfg = yaml.safe_load(f)
base = cfg.get("outputs", {}).get("base_dir", "outputs")
import pathlib
# base_dir is relative to the project root unless it's absolute
p = pathlib.Path(base)
if not p.is_absolute():
    p = pathlib.Path("${PROJECT_ROOT}") / p
print(str(p))
EOF
)"

echo "============================================================"
echo "  monoT5 Activation Patching — Smoke Test"
echo "  Config:  ${CONFIG}"
echo "  Outputs: ${OUTPUTS_DIR}"
echo "============================================================"
echo ""

# ---- Stage: find_data -------------------------------------------------------
echo "--- Stage: find_data ---"
bash "${SCRIPT_DIR}/find_data.sh" "${CONFIG}"
echo ""

# ---- Stage 00: prepare_pairs ------------------------------------------------
echo "--- Stage 00: prepare_pairs ---"
bash "${SCRIPT_DIR}/run_prepare_pairs.sh" "${CONFIG}"
echo ""

# ---- Stage 01: score_and_select ---------------------------------------------
echo "--- Stage 01: score_and_select ---"
bash "${SCRIPT_DIR}/run_score_and_select.sh" "${CONFIG}"
echo ""

# ---- Stage 02: cache_activations --------------------------------------------
echo "--- Stage 02: cache_activations ---"
bash "${SCRIPT_DIR}/run_cache_activations.sh" "${CONFIG}"
echo ""

# ---- Stage 03: layer_patching -----------------------------------------------
echo "--- Stage 03: run_layer_patching ---"
bash "${SCRIPT_DIR}/run_layer_patching.sh" "${CONFIG}"
echo ""

# ---- Stage 04: plot_results -------------------------------------------------
echo "--- Stage 04: plot_results ---"
bash "${SCRIPT_DIR}/run_plots.sh" "${CONFIG}"
echo ""

# ---- Verify that all expected output files exist ----------------------------
echo "--- Verifying output files ---"

EXPECTED_FILES=(
    "${OUTPUTS_DIR}/pairs/pairs.jsonl"
    "${OUTPUTS_DIR}/scores/all_scores.csv"
    "${OUTPUTS_DIR}/scores/selected_examples.jsonl"
    "${OUTPUTS_DIR}/patching/layer_patching_results.csv"
    "${OUTPUTS_DIR}/plots/attack_delta_hist.png"
    "${OUTPUTS_DIR}/plots/forward_layer_component_heatmap.png"
    "${OUTPUTS_DIR}/plots/reverse_layer_component_heatmap.png"
    "${OUTPUTS_DIR}/plots/combined_layer_component_heatmap.png"
)

ALL_OK=true
for f in "${EXPECTED_FILES[@]}"; do
    if [[ -f "${f}" ]]; then
        echo "  OK  ${f}"
    else
        echo "  MISSING  ${f}"
        ALL_OK=false
    fi
done

if [[ "${ALL_OK}" == "false" ]]; then
    echo ""
    echo "ERROR: One or more expected output files are missing."
    echo "Check the stage output above for error messages."
    exit 1
fi

echo ""
echo "All expected output files found."
echo ""

# ---- Sanity-check diagnostics -----------------------------------------------
echo "--- Stage 05: sanity_check_outputs ---"
bash "${SCRIPT_DIR}/run_sanity_check.sh" "${CONFIG}"
echo ""

echo "============================================================"
echo "  Smoke test complete. No errors."
echo "  Results in: ${OUTPUTS_DIR}"
echo "============================================================"
