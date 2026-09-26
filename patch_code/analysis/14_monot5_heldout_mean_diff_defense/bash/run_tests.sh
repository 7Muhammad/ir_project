#!/usr/bin/env bash
# bash/run_tests.sh — run the pytest suite. CPU-only, no GPU or full model
# download needed except for the position-mask tests (tokenizer only).
#
# Usage:
#   conda activate advseq2seq
#   bash bash/run_tests.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="$(dirname "$SCRIPT_DIR")"

cd "$EXP_DIR"
python3 -m pytest tests/ -v
