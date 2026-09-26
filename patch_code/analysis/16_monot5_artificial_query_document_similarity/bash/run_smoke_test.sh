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
         plots/fig1_clean_rho.png plots/fig7_genuine_vs_attack_gap.png plots/fig9_delta_sim_vs_delta_score.png; do
    if [[ -s "${EXP_DIR}/outputs_smoke/${f}" ]]; then echo "  OK       ${f}"; else echo "  MISSING  ${f}"; exit 1; fi
done
echo "=== Smoke test passed ==="
