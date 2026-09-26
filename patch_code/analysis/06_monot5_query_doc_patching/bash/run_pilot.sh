#!/usr/bin/env bash
# bash/run_pilot.sh — timing pilot (1 attack, n=5, all layers/conditions/regions).
# Run this BEFORE run_grid.sh; reports extrapolated full-grid runtime.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="$(dirname "$SCRIPT_DIR")"
python3 "${EXP_DIR}/scripts/01_run_pilot.py" --config "${EXP_DIR}/configs/pilot.yaml" "$@"
