#!/usr/bin/env bash
# bash/run_smoke.sh — end-to-end smoke test (2 attacks x 2 examples, CPU-ok).
# Writes to outputs_smoke/, never touches real outputs.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="$(dirname "$SCRIPT_DIR")"
bash "${SCRIPT_DIR}/run_all.sh" "${EXP_DIR}/configs/smoke.yaml" "$@"
