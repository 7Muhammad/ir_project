#!/usr/bin/env bash
# bash/run_smoke_test.sh — one attack, tiny sample sizes, outputs_smoke/,
# for quick debugging before launching the full 105-attack grid.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG="${EXP_DIR}/configs/smoke.yaml"

echo "=== Experiment 12 smoke test (config: $CONFIG) ==="
python3 "${EXP_DIR}/scripts/00_sanity_checks.py" --config "$CONFIG" --attack relevant_start_5 --n-examples 3
python3 "${EXP_DIR}/scripts/01_run_causal.py" --config "$CONFIG" --force
python3 "${EXP_DIR}/scripts/02_run_attention.py" --config "$CONFIG" --force
python3 "${EXP_DIR}/scripts/03_aggregate.py" --config "$CONFIG"
python3 "${EXP_DIR}/scripts/04_make_plots.py" --config "$CONFIG"
python3 "${EXP_DIR}/scripts/05_diagnostics.py" --config "$CONFIG"
echo "=== Smoke test done. See outputs_smoke/ ==="
