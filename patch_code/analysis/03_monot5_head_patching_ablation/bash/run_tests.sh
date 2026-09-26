#!/usr/bin/env bash
# bash/run_tests.sh — unit tests (tiny randomly-initialised T5, CPU, no downloads).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="$(dirname "$SCRIPT_DIR")"
cd "$EXP_DIR"
python3 -m pytest tests/ -v "$@"
