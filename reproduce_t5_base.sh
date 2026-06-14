#!/bin/bash
# reproduce_t5_base.sh — Phases 3 and 4 for monoT5-base
#
# Phases 1+2 (token creation and injection) were completed by reproduce_t5_small.sh.
# This script only scores the already-injected documents and computes rank changes.
#
# USAGE
#   bash reproduce_t5_base.sh
#   # Or via SLURM:
#   bash run_job.sh --job-name t5_base --gpu-type L40 --gpu-count 1 --command 'bash /home/ghoummaid/IR/reproduce_t5_base.sh'

set -euo pipefail

# =============================================================================
# CONFIGURATION
# =============================================================================

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/ecir24-adversarial-evaluation" && pwd)"

SCALE_SCORE_SCRIPT="$REPO_DIR/advseq2seq/stuffing/evaluation/scale_score.py"
RUN_SCALE_SCRIPT="$REPO_DIR/advseq2seq/stuffing/run_scale.py"
EVALUATE_SCRIPT="$REPO_DIR/advseq2seq/stuffing/evaluation/evaluate.py"

RUNS_DIR="$REPO_DIR/runs"
TOKEN_FILE="$RUNS_DIR/tokens_table3.txt"
INJECTED_DIR="$RUNS_DIR/injected"
SCORED_DIR="$RUNS_DIR/scored"
RANK_CHANGES_DIR="$RUNS_DIR/rank_changes"

MODEL_NAME="t5.base"
MODEL_CKPT="castorini/monot5-base-msmarco"
BATCH_SIZE=256

# =============================================================================
# SANITY CHECKS
# =============================================================================

echo "=== Sanity checks ==="

if [[ ! -f "$TOKEN_FILE" ]]; then
    echo "ERROR: $TOKEN_FILE not found."
    echo "  Run reproduce_t5_small.sh first to complete Phases 1+2."
    exit 1
fi

for dataset in dl19 dl20; do
    count=$(ls "$INJECTED_DIR/$dataset/" 2>/dev/null | wc -l | tr -d ' ')
    if [[ "$count" -lt 375 ]]; then
        echo "ERROR: Expected >=375 injected files in $INJECTED_DIR/$dataset/, found $count."
        echo "  Run reproduce_t5_small.sh first to complete Phases 1+2."
        exit 1
    fi
done
echo "  Injection files: OK"

if ! python -c "from pyterrier_t5 import MonoT5ReRanker" 2>/dev/null; then
    echo "ERROR: pyterrier-t5 is not installed. Run: pip install python-terrier pyterrier-t5"
    exit 1
fi

echo "=== GPU check ==="
GPU_INFO=$(python - << 'PYEOF'
import torch, sys
cuda_ok = torch.cuda.is_available()
torch_ver = torch.__version__
if cuda_ok:
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem  = torch.cuda.get_device_properties(0).total_memory // (1024**3)
    print(f"OK  torch={torch_ver}  GPU={gpu_name}  VRAM={gpu_mem}GB")
    sys.exit(0)
else:
    print(f"FAIL  torch={torch_ver}  CUDA not available")
    sys.exit(1)
PYEOF
)
GPU_EXIT=$?
echo "  $GPU_INFO"
if [[ $GPU_EXIT -ne 0 ]]; then
    echo "ERROR: CUDA not available. Scoring would fall back to CPU (~weeks of runtime)."
    echo "  Fix: pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu121"
    exit 1
fi

echo "  Model: $MODEL_CKPT"
echo "  OK"

# =============================================================================
# SETUP
# =============================================================================

mkdir -p \
    "$SCORED_DIR/dl19" \
    "$SCORED_DIR/dl20" \
    "$RANK_CHANGES_DIR/dl19" \
    "$RANK_CHANGES_DIR/dl20"

# =============================================================================
# PHASE 3 — Score attacked documents with monoT5-base
# =============================================================================

echo ""
echo "=== Phase 3: Scoring with $MODEL_NAME ($MODEL_CKPT) ==="
echo "  Slow step — expect ~30h on a single L40 GPU."
echo "  Files that already exist in the output directory are skipped (safe to resume)."

echo "  -> DL19"
python "$RUN_SCALE_SCRIPT" \
    --script "$SCALE_SCORE_SCRIPT" \
    --run_dir "$INJECTED_DIR/dl19/" \
    --output_dir "$SCORED_DIR/dl19/" \
    --name "$MODEL_NAME" \
    --batch_size "$BATCH_SIZE"

echo "  -> DL20"
python "$RUN_SCALE_SCRIPT" \
    --script "$SCALE_SCORE_SCRIPT" \
    --run_dir "$INJECTED_DIR/dl20/" \
    --output_dir "$SCORED_DIR/dl20/" \
    --name "$MODEL_NAME" \
    --batch_size "$BATCH_SIZE"

for dataset in dl19 dl20; do
    NORMAL_FILE="$SCORED_DIR/$dataset/normal_${MODEL_NAME}.tsv"
    if [[ ! -f "$NORMAL_FILE" ]]; then
        echo "ERROR: Baseline file missing: $NORMAL_FILE"
        echo "  Phase 3 may have crashed before scoring the first file."
        exit 1
    fi
done
echo "  Baseline files (normal_${MODEL_NAME}.tsv) confirmed present for both datasets."

# =============================================================================
# PHASE 4 — Compute rank changes
# =============================================================================

echo ""
echo "=== Phase 4: Computing rank changes ==="

for dataset in dl19 dl20; do
    echo "  -> $dataset"
    scored_path="$SCORED_DIR/$dataset"
    output_path="$RANK_CHANGES_DIR/$dataset"

    shopt -s nullglob
    for run_file in "$scored_path"/*_${MODEL_NAME}.tsv; do
        fname=$(basename "$run_file")
        if [[ "$fname" == normal_* ]]; then
            continue
        fi
        python "$EVALUATE_SCRIPT" \
            --run_file "$run_file" \
            --normal_dir "$scored_path" \
            --res_dump "$output_path"
    done
    shopt -u nullglob

    FILE_COUNT=$(ls "$output_path/"*_${MODEL_NAME}_rank_changes.tsv.gz 2>/dev/null | wc -l | tr -d ' ')
    echo "     $FILE_COUNT rank change files written to $output_path/ (expected 375)"
done

# =============================================================================
# SUMMARY
# =============================================================================

echo ""
echo "============================================================"
echo "  Done: $MODEL_NAME"
echo "============================================================"
echo ""
echo "  Rank change results:"
echo "    DL19: $(ls "$RANK_CHANGES_DIR/dl19/"*_${MODEL_NAME}_rank_changes.tsv.gz 2>/dev/null | wc -l | tr -d ' ') files in $RANK_CHANGES_DIR/dl19/"
echo "    DL20: $(ls "$RANK_CHANGES_DIR/dl20/"*_${MODEL_NAME}_rank_changes.tsv.gz 2>/dev/null | wc -l | tr -d ' ') files in $RANK_CHANGES_DIR/dl20/"
