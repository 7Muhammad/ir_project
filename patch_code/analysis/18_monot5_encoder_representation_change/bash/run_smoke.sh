#!/usr/bin/env bash
# bash/run_smoke.sh — 4 attacks x 24 instances into outputs_smoke/ (CPU, ~1.5 min); always --force.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="$(dirname "$SCRIPT_DIR")"
bash "${SCRIPT_DIR}/run_all.sh" "${EXP_DIR}/configs/smoke.yaml" --force
O="${EXP_DIR}/outputs_smoke"
for f in 01_repr_change/cka.csv 01_repr_change/checks.json 02_analysis/layer_summary.csv 02_analysis/correlations.csv \
         02_analysis/plots/fig1_cosine_by_layer.png 02_analysis/plots/fig3_linear_cka_by_layer.png; do
    if [[ -s "${O}/${f}" ]]; then echo "  OK       ${f}"; else echo "  MISSING  ${f}"; exit 1; fi
done
echo "=== Exp 18 smoke test passed ==="
