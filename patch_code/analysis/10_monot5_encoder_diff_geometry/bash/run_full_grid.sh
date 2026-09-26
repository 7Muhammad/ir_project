#!/usr/bin/env bash
# bash/run_full_grid.sh — optional extension: all 105 curated attacks, n=100
# (full depth, matching the primary run) -> outputs_full_grid/. Run
# bash/run_all.sh first.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="$(dirname "$SCRIPT_DIR")"
CFG="${EXP_DIR}/configs/full_grid.yaml"

python3 "${EXP_DIR}/scripts/01_extract_diffs.py" --config "$CFG" "$@"
python3 "${EXP_DIR}/scripts/02_pca_summary.py" --config "$CFG"
python3 "${EXP_DIR}/scripts/03_cross_attack_cosine.py" --config "$CFG"
python3 "${EXP_DIR}/scripts/04_position_concentration.py" --config "$CFG"
python3 "${EXP_DIR}/scripts/05_make_plots.py" --config "$CFG"
