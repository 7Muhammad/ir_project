#!/usr/bin/env bash
# bash/run_tests.sh — unit tests (real tokenizer for spans, tiny random T5 for hooks; CPU, no full-model downloads beyond the tokenizer).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="$(dirname "$SCRIPT_DIR")"
cd "$EXP_DIR"
python3 -m pytest tests/ -v "$@"
