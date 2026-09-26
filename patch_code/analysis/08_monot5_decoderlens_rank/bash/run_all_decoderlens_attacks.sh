#!/usr/bin/env bash
# bash/run_all_decoderlens_attacks.sh
# ===================================
# Run the full DecoderLens rank experiment over ALL discovered attacks:
#   discover -> (per attack) prepare -> score -> rank -> plot -> compare
#
# Usage:
#   conda activate advseq2seq
#   bash bash/run_all_decoderlens_attacks.sh [config_path] [extra args...]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

CONFIG="${1:-${PROJECT_ROOT}/configs/default.yaml}"
shift || true

if [[ ! -f "${CONFIG}" ]]; then
    echo "ERROR: Config file not found: ${CONFIG}"
    echo "Usage: bash bash/run_all_decoderlens_attacks.sh [path/to/config.yaml]"
    exit 1
fi

echo "============================================================"
echo "  monoT5 DecoderLens Rank Experiment (all attacks)"
echo "  Config: ${CONFIG}"
echo "============================================================"

echo ""
echo "--- Stage 01: discover attacks ---"
python "${PROJECT_ROOT}/scripts/01_discover_attacks.py" --config "${CONFIG}"

echo ""
echo "--- Stages 02-06: per-attack pipeline + comparison ---"
python "${PROJECT_ROOT}/scripts/07_run_all_decoderlens_attacks.py" \
    --config "${CONFIG}" "$@"

echo ""
echo "============================================================"
echo "  Done.  See outputs/attacks/ and outputs/attack_comparison/"
echo "============================================================"
