#!/usr/bin/env bash
# bash/run_debug.sh — Experiment 13: single-path debug run + required
# sanity checks. Run this BEFORE bash/run_sweep.sh.
#
# Usage:
#   conda activate advseq2seq
#   bash bash/run_debug.sh [config_path]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG="${1:-${EXP_DIR}/configs/default.yaml}"
CONFIG="$(realpath "$CONFIG")"

echo "=== Experiment 13: debug single-path run ==="
python3 "${EXP_DIR}/scripts/00_debug_single_path.py" --config "$CONFIG"

echo
echo "=== Experiment 13: required sanity checks ==="
python3 "${EXP_DIR}/scripts/01_sanity_checks.py" --config "$CONFIG"
