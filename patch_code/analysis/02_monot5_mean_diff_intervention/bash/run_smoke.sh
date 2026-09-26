#!/usr/bin/env bash
# bash/run_smoke.sh — run the full pipeline against configs/smoke.yaml
# (2 attacks x 3-5 examples, 3 flagged heads). CPU-friendly, a few minutes.
# Outputs go to outputs_smoke/ so real outputs are never touched.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="$(dirname "$SCRIPT_DIR")"

bash "${SCRIPT_DIR}/run_all.sh" "${EXP_DIR}/configs/smoke.yaml" "$@"
