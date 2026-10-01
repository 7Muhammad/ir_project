#!/usr/bin/env bash
# bash/run_paired_all_layers_smoke.sh — all-layers (144-head) scope on the paired smoke population.
# Reuses the smoke paired manifest (runs bash/run_paired_smoke.sh first if it is missing); writes
# outputs_paired_smoke/*_all_layers. Stage 18 is forced; stage 17 is reused, never rebuilt.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="$(dirname "$SCRIPT_DIR")"
PY="/home/ghoummaid/miniconda3/envs/advseq2seq/bin/python"
CFG="${EXP_DIR}/configs/paired_all_layers_smoke.yaml"
O="${EXP_DIR}/outputs_paired_smoke"
[[ -s "${O}/17_paired_manifest/status.json" ]] || bash "${SCRIPT_DIR}/run_paired_smoke.sh"
cd "$EXP_DIR"
echo "=== Experiment 16 PAIRED ALL-LAYERS SMOKE TEST ==="
"$PY" scripts/18_run_paired_forward.py --config "$CFG" --force
"$PY" scripts/19_paired_anomaly.py --config "$CFG"
"$PY" scripts/20_paired_detector.py --config "$CFG"
"$PY" scripts/21_plot_paired.py --config "$CFG"
for f in 18_paired_forward_all_layers/per_attack_status.csv 19_paired_anomaly_all_layers/paired_instances.csv.gz \
         19_paired_anomaly_all_layers/layer_summary.csv 20_paired_detector_all_layers/topk_detector_metrics.csv \
         21_paired_report_all_layers/final_summary.json plots/paired_all_layers/fig_paired_heads_all_layers.png \
         plots/paired_all_layers/fig_paired_abnormal_count_distribution.png plots/paired_all_layers/fig_paired_abnormal_count_change.png \
         plots/paired_all_layers/fig_paired_head_transitions.png plots/paired_all_layers/fig_paired_per_head_abnormal_rate.png \
         plots/paired_all_layers/fig_paired_low_high.png plots/paired_all_layers/fig_paired_topk_detector.png \
         plots/paired_all_layers/fig_paired_layer_summary.png plots/paired_all_layers/fig_paired_scatter_top_heads.png; do
    if [[ -s "${O}/${f}" ]]; then echo "  OK       ${f}"; else echo "  MISSING  ${f}"; exit 1; fi
done
echo "=== Paired all-layers smoke test passed ==="
