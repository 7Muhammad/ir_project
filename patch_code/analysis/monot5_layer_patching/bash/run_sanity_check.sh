#!/usr/bin/env bash
# bash/run_sanity_check.sh
# ========================
# Run the sanity-check diagnostic script on the experiment outputs.
#
# This script reads the CSV and JSONL files produced by Stages 00–03 and
# prints statistics to help you judge whether the experiment looks correct.
#
# It is safe to run multiple times — it does not modify any files.
#
# Usage:
#   conda activate advseq2seq
#   bash bash/run_sanity_check.sh [config_path]
#
# Default config: configs/default.yaml
#
# Examples:
#   # Check smoke-test outputs
#   bash bash/run_sanity_check.sh configs/smoke.yaml
#
#   # Check real experiment outputs
#   bash bash/run_sanity_check.sh configs/default.yaml

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

CONFIG="${1:-${PROJECT_ROOT}/configs/default.yaml}"

if [[ ! -f "${CONFIG}" ]]; then
    echo "ERROR: Config file not found: ${CONFIG}"
    echo "Usage: bash bash/run_sanity_check.sh [path/to/config.yaml]"
    exit 1
fi

if ! python3 -c "import torch" 2>/dev/null; then
    echo "ERROR: Required packages not found. Activate the conda env first:"
    echo "  conda activate advseq2seq"
    exit 1
fi

python3 "${PROJECT_ROOT}/scripts/05_sanity_check_outputs.py" --config "${CONFIG}"
