#!/usr/bin/env bash
# bash/run_smoke.sh — first 6 stage-22 base pairs (150 sequences) into outputs_smoke/; always --force. CPU-feasible.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="$(dirname "$SCRIPT_DIR")"
bash "${SCRIPT_DIR}/run_all.sh" "${EXP_DIR}/configs/smoke.yaml" --force
O="${EXP_DIR}/outputs_smoke"
for f in 01_attention_stats/attention_statistics.npz 01_attention_stats/validation.json 02_analysis/summary_table.md \
         02_analysis/injected_table.md 02_analysis/plots/D_injected_mass_heatmaps.png; do
    if [[ -s "${O}/${f}" ]]; then echo "  OK       ${f}"; else echo "  MISSING  ${f}"; exit 1; fi
done
echo "=== Exp 17 smoke test passed ==="
