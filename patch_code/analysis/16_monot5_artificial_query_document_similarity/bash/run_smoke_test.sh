#!/usr/bin/env bash
# bash/run_smoke_test.sh — the whole pipeline on configs/smoke.yaml (5 clean pairs,
# 2 qrel queries x 3 docs/class, 2 attacks x 5 examples, all 25 checkpoints,
# fresh-vs-cached score check), written to outputs_smoke/. Always --force. CPU-feasible.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="$(dirname "$SCRIPT_DIR")"
echo "=== Experiment 16 SMOKE TEST ==="
bash "${SCRIPT_DIR}/run_all.sh" "${EXP_DIR}/configs/smoke.yaml" --force
echo ""
echo "=== Checking expected smoke outputs ==="
for f in manifests/clean.jsonl manifests/attacks.jsonl.gz manifests/qrel_balanced.jsonl \
         manifests/qrel_query_summary.csv manifests/provenance.json \
         01_clean/clean_similarity.csv 01_clean/sanity_report.json 02_qrel/qrel_similarity.csv.gz \
         03_attack/attack_similarity.csv.gz 04_analysis/clean_correlation.csv \
         04_analysis/qrel_per_query_trajectory.csv 04_analysis/genuine_trajectory.csv \
         04_analysis/attack_per_attack_means.csv 04_analysis/attack_global_trajectory.csv \
         04_analysis/attack_accumulation.csv 04_analysis/attack_signflip_fdr.csv \
         04_analysis/delta_sim_vs_delta_score.csv 04_analysis/genuine_vs_attack.csv \
         04_analysis/attack_breakdowns.csv 04_analysis/summary.json \
         plots/fig1_clean_rho.png plots/fig7_genuine_vs_attack_gap.png plots/fig9_delta_sim_vs_delta_score.png \
         07_decoder/qrel_decoder.csv.gz 07_decoder/attack_decoder.csv.gz \
         plots/fig_successful_attacks_vs_genuine_similarity.png plots/fig_decoder_message_similarity_four_populations.png \
         09_heads/qrel_heads.csv.gz 09_heads/attack_heads.csv.gz 10_head_analysis/head_similarity_encoder.csv \
         10_head_analysis/head_similarity_decoder.csv plots/fig_heads_encoder_four_populations.png \
         plots/fig_heads_decoder_four_populations.png plots/fig_heads_decoder_four_populations_by_layer.png \
         11_decoder_probe/qrel_decoder_probe.csv.gz 11_decoder_probe/attack_decoder_probe.csv.gz \
         plots/fig_decoder_probe_similarity_four_populations.png; do
    if [[ -s "${EXP_DIR}/outputs_smoke/${f}" ]]; then echo "  OK       ${f}"; else echo "  MISSING  ${f}"; exit 1; fi
done
echo "=== Smoke test passed ==="
