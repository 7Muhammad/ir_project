#!/usr/bin/env bash
# bash/run_smoke_test.sh — the whole pipeline on configs/smoke.yaml (3 attacks:
# start / end / scattered-random, 2 heads in 2 layers, 3 examples/attack),
# written to outputs_smoke/. Always --force. CPU-feasible (a few minutes).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="$(dirname "$SCRIPT_DIR")"
echo "=== Experiment 15 SMOKE TEST ==="
bash "${SCRIPT_DIR}/run_all.sh" "${EXP_DIR}/configs/smoke.yaml" --force
echo ""
echo "=== Checking expected smoke outputs ==="
for f in sample_manifest.jsonl 00_samples/sample_summary.csv 01_sanity/sanity_report.json \
         05_aggregate/raw_all_query_edges.csv 05_aggregate/raw_single_query_edges.csv.gz \
         05_aggregate/raw_whole_head.csv 05_aggregate/per_attack_head_all_query.csv \
         05_aggregate/global_head_summary.csv 05_aggregate/edge_vs_whole_head.csv \
         05_aggregate/token_summary_by_head.csv 06_statistics/head_statistics.csv \
         plots/fig1_edge_combined_by_head.png plots/fig7_token_examples.png; do
    if [[ -s "${EXP_DIR}/outputs_smoke/${f}" ]]; then echo "  OK       ${f}"; else echo "  MISSING  ${f}"; exit 1; fi
done
echo "=== Smoke test passed ==="
