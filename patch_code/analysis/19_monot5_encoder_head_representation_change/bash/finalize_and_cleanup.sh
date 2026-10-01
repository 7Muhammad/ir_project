#!/usr/bin/env bash
# bash/finalize_and_cleanup.sh — unattended: stage 02 (analysis + figures), then delete the ~2.1 GB of
# stage-01 intermediates (metrics/*.npy, cka_stats.npz) ONLY if stage 02 succeeded and every kept output exists.
# (User decision 2026-09-30: keep only the ~20 MB of tables / figures.)
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="$(dirname "$SCRIPT_DIR")"
PY=/home/ghoummaid/miniconda3/envs/advseq2seq/bin/python
cd "$EXP_DIR"
S1=outputs/01_head_change
S2=outputs/02_analysis
grep -q '"success"' "$S1/status.json" || { echo "stage 01 not successful -> abort (nothing deleted)"; exit 1; }
echo "=== [$(date '+%F %T')] Exp 19 stage 02 ==="
"$PY" scripts/02_analyze_head_change.py
grep -q '"success"' "$S2/status.json" || { echo "stage 02 not successful -> nothing deleted"; exit 1; }
for f in head_summary.csv head_correlations.csv cka.csv causal_comparison.csv causal_head_ranks.csv topk_heads.csv \
         layer_concentration.csv success_contrast.csv relevance_contrast.csv summary.json \
         plots/fig1_one_minus_cosine_heads.png plots/fig2_normalized_l2_heads.png plots/fig3_one_minus_cka_heads.png \
         plots/fig3b_one_minus_cka_query_centred_heads.png plots/fig4_causal_vs_other_heads.png \
         plots/fig5_top_divergent_heads.png plots/fig6_layer_transition_heads.png \
         plots/fig7_success_contrast_and_delta_score.png; do
    [[ -s "$S2/$f" ]] || { echo "missing $S2/$f -> nothing deleted"; exit 1; }
done
[[ -s "$S1/cka.csv" && -s "$S1/instances.parquet" ]] || { echo "missing stage-01 tables -> nothing deleted"; exit 1; }
echo "=== [$(date '+%F %T')] all outputs present; disk before cleanup ==="
du -sh outputs
rm -rf "$S1/metrics" "$S1/cka_stats.npz"
echo "=== [$(date '+%F %T')] deleted $S1/metrics and $S1/cka_stats.npz; disk after ==="
du -sh outputs
echo "=== done ==="
