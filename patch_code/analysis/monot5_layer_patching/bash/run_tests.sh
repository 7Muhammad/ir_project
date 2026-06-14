#!/usr/bin/env bash
# bash/run_tests.sh
# =================
# Run all unit tests with pytest.
#
# Usage:
#   conda activate advseq2seq
#   bash bash/run_tests.sh
#
# What the tests check:
#   test_patching_math.py   — effect formulas (no model needed, very fast)
#   test_tokenization.py    — "true"/"false" are single tokens; alignment
#   test_padded_control.py  — padded-control construction invariants
#   test_scoring.py         — monoT5 scoring returns a finite float
#   test_hooks.py           — cache/patch hooks fire and are cleaned up
#
# Tests that load the model (test_scoring.py, test_hooks.py, test_tokenization.py,
# test_padded_control.py) will download or use the cached checkpoint.
# First run takes ~30 s; subsequent runs are fast (cached model).
#
# To run only the fast math tests (no model):
#   pytest tests/test_patching_math.py -v

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "${PROJECT_ROOT}"

# ---- Validate environment --------------------------------------------------
echo "============================================================"
echo "  monoT5 Activation Patching — Unit Tests"
echo "============================================================"
echo ""

if ! python3 -c "import pytest" 2>/dev/null; then
    echo "ERROR: pytest is not installed in the current Python environment."
    echo "Install it with:  pip install pytest"
    echo "Or activate the correct conda env:  conda activate advseq2seq"
    exit 1
fi

if ! python3 -c "import torch" 2>/dev/null; then
    echo "ERROR: torch is not available. Activate the conda env first:"
    echo "  conda activate advseq2seq"
    exit 1
fi

# ---- Run tests -------------------------------------------------------------
echo "Running: pytest tests/ -v"
echo ""

# -v  = verbose output (show each test name and PASSED/FAILED)
# --tb=short  = short tracebacks on failure (easier to read than full stack)
pytest tests/ -v --tb=short

echo ""
echo "============================================================"
echo "  All tests passed."
echo "============================================================"
