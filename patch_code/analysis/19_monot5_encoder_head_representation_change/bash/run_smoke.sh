#!/usr/bin/env bash
# bash/run_smoke.sh — 4 attacks x 24 instances into outputs_smoke/ (CPU, ~2 min); always --force.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="$(dirname "$SCRIPT_DIR")"
bash "${SCRIPT_DIR}/run_all.sh" "${EXP_DIR}/configs/smoke.yaml" --force
O="${EXP_DIR}/outputs_smoke"
for f in 01_head_change/cka.csv 01_head_change/checks.json 02_analysis/head_summary.csv 02_analysis/causal_comparison.csv \
         02_analysis/plots/fig1_one_minus_cosine_heads.png 02_analysis/plots/fig4_causal_vs_other_heads.png; do
    if [[ -s "${O}/${f}" ]]; then echo "  OK       ${f}"; else echo "  MISSING  ${f}"; exit 1; fi
done
echo "=== Exp 19 smoke test passed ==="
