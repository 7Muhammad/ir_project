#!/bin/bash
# reproduce_table3.sh — Phase B: Aggregate rank changes and generate LaTeX Table 3
#
# Run this AFTER all three SLURM jobs (t5_base, t5_large, t5_3b) have finished.
#
# USAGE
#   bash reproduce_table3.sh
#
# OUTPUT
#   aggregated.tsv  — intermediate summary (600 rows)
#   table3.tex      — LaTeX Table 3, ready to include in the paper

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$SCRIPT_DIR/ecir24-adversarial-evaluation"
RANK_CHANGES_DIR="$REPO_DIR/runs/rank_changes"
AGGREGATE_SCRIPT="$SCRIPT_DIR/aggregate_rank_changes.py"
TABLE_SCRIPT="$REPO_DIR/advseq2seq/stuffing/table_generation/make_scale_table.py"
AGGREGATED_TSV="$SCRIPT_DIR/aggregated.tsv"
TABLE_TEX="$SCRIPT_DIR/table3.tex"

# =============================================================================
# SANITY CHECK — verify all 4 models have completed rank_changes
# =============================================================================

echo "=== Checking rank change files ==="
ALL_OK=1
for model in t5.small t5.base t5.large t5.3b; do
    for dataset in dl19 dl20; do
        count=$(ls "$RANK_CHANGES_DIR/$dataset/"*_${model}_rank_changes.tsv.gz 2>/dev/null | wc -l | tr -d ' ')
        if [[ "$count" -lt 375 ]]; then
            echo "  MISSING  $model / $dataset: $count/375 files (job may still be running)"
            ALL_OK=0
        else
            echo "  OK       $model / $dataset: $count files"
        fi
    done
done

if [[ "$ALL_OK" -ne 1 ]]; then
    echo ""
    echo "ERROR: Not all models are complete. Wait for remaining SLURM jobs to finish."
    exit 1
fi

# =============================================================================
# STEP 1 — Aggregate rank changes → aggregated.tsv
# =============================================================================

echo ""
echo "=== Step 1: Aggregating rank changes ==="
python "$AGGREGATE_SCRIPT" \
    --rank_changes_dir "$RANK_CHANGES_DIR" \
    --out_file "$AGGREGATED_TSV"

echo "  Written: $AGGREGATED_TSV"

# =============================================================================
# STEP 2 — Generate LaTeX Table 3
# =============================================================================

echo ""
echo "=== Step 2: Generating LaTeX Table 3 ==="
python "$TABLE_SCRIPT" \
    --run_file "$AGGREGATED_TSV" \
    --out_file "$TABLE_TEX"

echo "  Written: $TABLE_TEX"

echo ""
echo "============================================================"
echo "  Done. Include table3.tex in your paper with:"
echo "    \input{table3}"
echo "============================================================"
