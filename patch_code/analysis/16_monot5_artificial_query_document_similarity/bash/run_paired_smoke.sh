#!/usr/bin/env bash
# bash/run_paired_smoke.sh — paired extension on configs/paired_smoke.yaml (4 attacks, 15 queries,
# <= 4 base pairs per group per query) into outputs_paired_smoke/. Always --force. CPU-feasible.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="$(dirname "$SCRIPT_DIR")"
echo "=== Experiment 16 PAIRED SMOKE TEST ==="
bash "${SCRIPT_DIR}/run_paired.sh" "${EXP_DIR}/configs/paired_smoke.yaml" --force
echo ""
echo "=== Checking expected smoke outputs ==="
O="${EXP_DIR}/outputs_paired_smoke"
for f in 17_paired_manifest/base_pairs.jsonl.gz 17_paired_manifest/fold_map.json 17_paired_manifest/alignment_failures.csv \
         17_paired_manifest/provenance.json 18_paired_forward/per_attack_status.csv \
         19_paired_anomaly/paired_instances.csv.gz 19_paired_anomaly/reference_fits.csv 19_paired_anomaly/per_head_paired.csv \
         19_paired_anomaly/paired_count_summary.csv 19_paired_anomaly/document_level.csv.gz \
         19_paired_anomaly/document_level_summary.csv 19_paired_anomaly/low_high_summary.csv \
         20_paired_detector/topk_detector_metrics.csv 20_paired_detector/detector_summary.json \
         21_paired_report/final_summary.json 21_paired_report/old_outputs_check.json \
         plots/paired/fig_paired_layer_similarity.png plots/paired/fig_paired_heads_L9_L11.png \
         plots/paired/fig_paired_abnormal_count_distribution.png plots/paired/fig_paired_abnormal_count_change.png \
         plots/paired/fig_paired_head_transitions.png plots/paired/fig_paired_per_head_abnormal_rate.png \
         plots/paired/fig_paired_low_high.png plots/paired/fig_paired_topk_detector.png \
         plots/paired/fig_paired_scatter_top_heads.png plots/paired/fig_paired_document_level.png; do
    if [[ -s "${O}/${f}" ]]; then echo "  OK       ${f}"; else echo "  MISSING  ${f}"; exit 1; fi
done
echo "=== Paired smoke test passed ==="
