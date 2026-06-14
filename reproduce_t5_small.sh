#!/bin/bash
# =============================================================================
# reproduce_t5_small.sh
#
# Reproduces the monoT5-small keyword-stuffing attack results from:
#   "Analyzing Adversarial Attacks on Sequence-to-Sequence Relevance Models"
#   ECIR 2024 — Parry, Fröbe, MacAvaney, Potthast, Hagen
#   https://arxiv.org/pdf/2403.07654
#
# WHAT THIS SCRIPT PRODUCES
# --------------------------
#   For every combination of:
#     - 25 attack tokens (prompt tokens, control tokens, synonyms, sub-words, misspellings)
#     - 3 injection positions (start, end, random)
#     - 5 repetition counts (n=1..5)
#     - 2 datasets (TREC DL 2019 and 2020)
#   it computes per-document rank changes when monoT5-small re-ranks the
#   attacked document instead of the original.
#
#   Final outputs (in runs/rank_changes/dl{19,20}/):
#     {token}_{mode}_{n}_bm25_{year}_t5.small_rank_changes.tsv.gz
#   Each file contains: qid, docno, rank_change, score_change, success, old_rank, new_rank
#
# PIPELINE OVERVIEW
# -----------------
#   Phase 1: Create tokens_table3.txt (25 tokens across 5 groups)
#   Phase 2: Inject tokens into BM25 candidate documents  → 375 TSV files per dataset
#   Phase 3: Score attacked documents with monoT5-small   → 375 scored TSV files per dataset
#            (also creates normal_t5.small.tsv — the unattacked baseline scores)
#   Phase 4: Compute rank changes using baseline          → 375 rank-change .tsv.gz per dataset
#
# WHY evaluate.py INSTEAD OF compute_rank_change.py
# --------------------------------------------------
#   scale_score.py (Phase 3) scores the attacked text and puts the result in
#   'augmented_score'. It only scores the original text ('text_0') for the
#   FIRST file it processes in an output directory, saving those scores to
#   normal_t5.small.tsv. For all subsequent files, the 'score' column is
#   WRONG (it equals augmented_score because the reranker overwrites it).
#
#   compute_rank_change.py reads 'score' and 'augmented_score' from the same
#   TSV, so it would use the wrong baseline for all but the first file.
#
#   evaluate.py fixes this: it reads the baseline from normal_t5.small.tsv
#   (via --normal_dir) and uses that as the reference ranking, ignoring the
#   stale 'score' column. It matches the right baseline by checking whether
#   't5.small' appears in the run file path.
#
# WHY join_all_scale.py IS SKIPPED
# ---------------------------------
#   join_all_scale.py would re-attach the correct baseline 'score' to every
#   file so that compute_rank_change.py could be used. But it requires
#   normal files for all four monoT5 sizes simultaneously (small, base, large,
#   3B) and crashes if any are missing. Since we are only running t5.small,
#   using evaluate.py with --normal_dir is the simpler and correct approach.
#
# REQUIREMENTS
# ------------
#   conda activate <your_env>
#   pip install python-terrier pyterrier-t5 pandas numpy tqdm fire torch transformers
#   (GPU strongly recommended — Phase 3 scores ~750 files through monoT5-small)
#
# USAGE
#   bash reproduce_t5_small.sh
#
#   To resume after interruption: re-run the script. Scoring (Phase 3) and rank
#   changes (Phase 4) both skip files that already exist in the output directory.
# =============================================================================

set -euo pipefail

# =============================================================================
# CONFIGURATION
# =============================================================================

# Path to the cloned repo (assumed to be alongside this script)
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/ecir24-adversarial-evaluation" && pwd)"

# Key scripts inside the repo
# NOTE: the file names in the repo are counterintuitive — the roles are:
#   token_injection.py     = the ORCHESTRATOR (loops n/modes, calls run_token_injection.py via subprocess)
#   run_token_injection.py = the WORKER (Syringe class, reads TSV, writes attacked TSV files)
INJECTION_ORCHESTRATOR="$REPO_DIR/advseq2seq/stuffing/injection/token_injection.py"
INJECTION_WORKER="$REPO_DIR/advseq2seq/stuffing/injection/run_token_injection.py"
SCALE_SCORE_SCRIPT="$REPO_DIR/advseq2seq/stuffing/evaluation/scale_score.py"
RUN_SCALE_SCRIPT="$REPO_DIR/advseq2seq/stuffing/run_scale.py"
EVALUATE_SCRIPT="$REPO_DIR/advseq2seq/stuffing/evaluation/evaluate.py"

# Input data
DATA_DIR="$REPO_DIR/data"
BM25_19="$DATA_DIR/bm25_19.tsv.gz"   # BM25 top-1000 for TREC DL 2019
BM25_20="$DATA_DIR/bm25_20.tsv.gz"   # BM25 top-1000 for TREC DL 2020

# Working directories (created under the repo's runs/ folder)
RUNS_DIR="$REPO_DIR/runs"
TOKEN_FILE="$RUNS_DIR/tokens_table3.txt"
INJECTED_DIR="$RUNS_DIR/injected"
SCORED_DIR="$RUNS_DIR/scored"
RANK_CHANGES_DIR="$RUNS_DIR/rank_changes"

# monoT5-small settings
MODEL_NAME="t5.small"                               # used as --name and in output filenames
MODEL_CKPT="castorini/monot5-small-msmarco-100k"    # HuggingFace checkpoint
BATCH_SIZE=256                                      # default batch size for t5.small per run_scale.py

# =============================================================================
# SANITY CHECKS
# =============================================================================

echo "=== Sanity checks ==="

if [[ ! -d "$REPO_DIR" ]]; then
    echo "ERROR: Repo not found at $REPO_DIR"
    echo "  Expected: git clone https://github.com/Parry-Parry/ecir24-adversarial-evaluation"
    echo "  in the same directory as this script"
    exit 1
fi

if [[ ! -f "$BM25_19" ]] || [[ ! -f "$BM25_20" ]]; then
    echo "ERROR: BM25 data files not found in $DATA_DIR"
    echo "  Expected: bm25_19.tsv.gz and bm25_20.tsv.gz"
    exit 1
fi

if ! python -c "import pyterrier" 2>/dev/null; then
    echo "ERROR: python-terrier is not installed. Run: pip install python-terrier pyterrier-t5"
    exit 1
fi

if ! python -c "from pyterrier_t5 import MonoT5ReRanker" 2>/dev/null; then
    echo "ERROR: pyterrier-t5 is not installed. Run: pip install pyterrier-t5"
    exit 1
fi

# ---------------------------------------------------------------------------
# GPU check — hard fail if CUDA is not available.
#
# scale_score.py passes device=torch.device('cuda' if torch.cuda.is_available()
# else 'cpu') to the reranker. If CUDA is not available it silently falls back
# to CPU, which makes each file take ~38 minutes instead of ~1-2 minutes —
# leading to an estimated 19 days of runtime instead of ~2 hours.
#
# Common causes if this check fails despite requesting a GPU in SLURM:
#   1. torch was installed as the CPU-only build (+cpu suffix).
#      Fix: pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu121
#   2. The SLURM job did not actually get allocated a GPU.
#      Fix: check `squeue -u $USER` and verify the GRES column shows a GPU.
#   3. CUDA driver version on the node is older than the torch build requires.
#      Fix: use a matching torch wheel (cu118, cu117, etc.) in setup_env.sh.
# ---------------------------------------------------------------------------
echo ""
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
    echo ""
    echo "ERROR: CUDA is not available. Scoring would fall back to CPU and take ~19 days."
    echo "  torch version installed: $(python -c 'import torch; print(torch.__version__)')"
    echo ""
    echo "  To fix — reinstall torch with the correct CUDA build:"
    echo "    pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu121"
    echo ""
    echo "  Then resubmit the SLURM job."
    exit 1
fi

echo "  Repo:    $REPO_DIR"
echo "  Data:    $BM25_19, $BM25_20"
echo "  Model:   $MODEL_CKPT"
echo "  Outputs: $RUNS_DIR"
echo "  OK"

# =============================================================================
# SETUP — Create working directories
# =============================================================================

echo ""
echo "=== Setup: Creating working directories ==="

mkdir -p \
    "$INJECTED_DIR/dl19" \
    "$INJECTED_DIR/dl20" \
    "$SCORED_DIR/dl19" \
    "$SCORED_DIR/dl20" \
    "$RANK_CHANGES_DIR/dl19" \
    "$RANK_CHANGES_DIR/dl20"

echo "  Created under $RUNS_DIR/"

# =============================================================================
# PHASE 1 — Create tokens_table3.txt
#
# These are the 25 attack tokens from Table 3 of the paper, grouped as:
#   Prompt Tokens   — tokens that appear in the monoT5 prompt itself
#   Control Tokens  — unrelated tokens (negative control)
#   Synonyms        — words semantically related to "relevant"
#   Sub-Words       — morphological variants of "relevant"
#   Misspellings    — misspelled variants
#
# token_injection.py strips colons and spaces when building output filenames:
#   "relevant: true" → output file named "relevanttrue_start_3_bm25_19.tsv"
#   "information:"   → output file named "information_start_3_bm25_19.tsv"
# =============================================================================

echo ""
echo "=== Phase 1: Creating tokens_table3.txt ==="

# tokens_table3.txt — the 25 attack tokens used in Table 3 of the paper.
#
# token_injection.py normalizes each token for use in output filenames by
# stripping colons and spaces:  "relevant: true" → "relevanttrue"
# make_scale_table.py uses these normalized names to match result files,
# so every token here must be present or the table script will crash.
#
# Group 1: Prompt Tokens — appear verbatim in the monoT5 prompt template
#   ("Query: q Document: d Relevant:"). Injecting these is the core attack.
cat > "$TOKEN_FILE" << 'EOF'
true
false
relevant:
relevant: true
relevant: false
EOF

# Group 2: Control Tokens — unrelated words used as negative controls
# to show the attack is specific to prompt-like tokens, not just any words.
cat >> "$TOKEN_FILE" << 'EOF'
bar
baz
information:
information: bar
information: baz
relevant: bar
information: true
EOF

# Group 3: Synonyms — words semantically similar to "relevant".
# Tests whether monoT5 responds to meaning or to the literal token string.
cat >> "$TOKEN_FILE" << 'EOF'
pertinent
significant
related
associated
important
EOF

# Group 4: Sub-Words — morphological variants of "relevant".
# Tests whether partial lexical overlap with the prompt token matters.
cat >> "$TOKEN_FILE" << 'EOF'
relevancy
relevance
relevantly
irrelevant
EOF

# Group 5: Misspellings — intentionally misspelled variants.
# NOTE: ChatGPT's description of the token list omits this group,
# but it is present in make_scale_table.py's TOKEN_GROUPS and is
# required for Table 3. Without it, make_scale_table.py will crash.
cat >> "$TOKEN_FILE" << 'EOF'
relevanty
relevent
trues
falses
EOF

TOKEN_COUNT=$(wc -l < "$TOKEN_FILE" | tr -d ' ')
echo "  Written $TOKEN_COUNT tokens to $TOKEN_FILE"

# =============================================================================
# PHASE 2 — Inject tokens into BM25 documents
#
# run_token_injection.py (orchestrator) calls token_injection.py (worker) for
# every combination of:
#   n    in {1, 2, 3, 4, 5}      — number of token repetitions
#   mode in {start, end, random} — injection position
#
# Total: 25 tokens × 3 modes × 5 repetitions = 375 files per dataset.
#
# Output file format:
#   {tokenname}_{mode}_{n}_bm25_{year}.tsv
#
# Output columns (TSV):
#   qid, query, docno, score, rank, text_0, text
#   text_0 = original document text
#   text    = attacked document text (token repeated n times, injected at mode)
#
# Example (token="relevant:", mode=start, n=3):
#   text = "relevant: relevant: relevant: [original document text]"
#
# Note: pandas can read .tsv.gz directly, so the input bm25_19.tsv.gz does
# not need to be decompressed first.
# =============================================================================

echo ""
echo "=== Phase 2: Injecting tokens into documents ==="

echo "  -> DL19 (bm25_19.tsv.gz)"
python "$INJECTION_ORCHESTRATOR" \
    --script "$INJECTION_WORKER" \
    --token_file "$TOKEN_FILE" \
    --doc_file "$BM25_19" \
    --output_dir "$INJECTED_DIR/dl19/"

DL19_INJECTED=$(ls "$INJECTED_DIR/dl19/" | wc -l | tr -d ' ')
echo "  DL19: $DL19_INJECTED files created (expected 375)"

echo "  -> DL20 (bm25_20.tsv.gz)"
python "$INJECTION_ORCHESTRATOR" \
    --script "$INJECTION_WORKER" \
    --token_file "$TOKEN_FILE" \
    --doc_file "$BM25_20" \
    --output_dir "$INJECTED_DIR/dl20/"

DL20_INJECTED=$(ls "$INJECTED_DIR/dl20/" | wc -l | tr -d ' ')
echo "  DL20: $DL20_INJECTED files created (expected 375)"

# =============================================================================
# PHASE 3 — Score attacked documents with monoT5-small
#
# run_scale.py loops every file in the injected directory and calls
# scale_score.py for each, using the monoT5-small checkpoint.
#
# scale_score.py behavior per file:
#   1. Reads the attacked TSV (columns: qid, docno, text_0, text, ...)
#   2. Scores 'text' (attacked document) → result stored as 'augmented_score'
#   3. FIRST FILE ONLY: also scores 'text_0' (original) → saves as
#      'normal_t5.small.tsv' in the output directory. This is the shared
#      unattacked baseline used in Phase 4.
#   4. Writes output TSV with augmented_score added.
#
# Output filename: {original_filename}_t5.small.tsv
#
# IMPORTANT — 'score' column in output files:
#   The first file output has correct 'score' (original) and 'augmented_score'
#   (attacked). For all subsequent files, 'score' incorrectly equals
#   'augmented_score' because the reranker overwrites the score column.
#   Phase 4 (evaluate.py) reads the correct baseline from normal_t5.small.tsv
#   directly, so this stale 'score' column does not affect the final results.
#
# NOTE: run_scale.py scans all files in the directory with no extension filter.
# The injected directories should contain only .tsv files (no other files).
#
# GPU REQUIRED: scoring 375 files through monoT5-small is slow on CPU.
# Expect several hours on a single GPU for both datasets combined.
# Files that already exist in the output directory are skipped (safe to resume).
# =============================================================================

echo ""
echo "=== Phase 3: Scoring with monoT5-small ($MODEL_CKPT) ==="
echo "  This is the slow step — may take several hours on GPU."

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

# Verify that the baseline file was created by Phase 3 before proceeding.
# If scoring crashed on the first file, normal_t5.small.tsv will be missing
# and Phase 4 will fail with a confusing FileNotFoundError.
for dataset in dl19 dl20; do
    NORMAL_FILE="$SCORED_DIR/$dataset/normal_${MODEL_NAME}.tsv"
    if [[ ! -f "$NORMAL_FILE" ]]; then
        echo "ERROR: Baseline file missing: $NORMAL_FILE"
        echo "  Phase 3 may have crashed before scoring the first file."
        echo "  Check for errors above, then re-run this script."
        exit 1
    fi
done
echo "  Baseline files (normal_t5.small.tsv) confirmed present for both datasets."

# =============================================================================
# PHASE 4 — Compute rank changes
#
# For each scored file, evaluate.py computes per-document rank change when
# the attacked document's score ('augmented_score') replaces its original
# score in the ranking.
#
# evaluate.py is called with --normal_dir pointing to the scored directory,
# where normal_t5.small.tsv lives. It auto-selects that file by detecting
# 't5.small' in the run file path.
#
# Metrics computed per document:
#   rank_change  = old_rank - new_rank  (positive = attack improved rank)
#   success      = rank_change > 0
#   old_rank     = document's rank in the unattacked baseline
#   new_rank     = document's rank after the attack
#   score_change = augmented_score - score  (note: 'score' may be stale —
#                  this metric is unreliable but rank_change/success are correct)
#
# Output: {original_name}_rank_changes.tsv.gz  (one per attacked file)
#
# NOTE: run_evaluate.py is NOT used here because it only passes --run_file and
# --res_dump to the evaluation script — it does not support --normal_dir.
# We loop manually to pass all three arguments.
#
# NOTE: evaluate.py skips files whose output already exists, but the existence
# check assumes .tsv.gz input. For .tsv input the check is ineffective, so
# already-processed files will be re-processed if you re-run this script.
# =============================================================================

echo ""
echo "=== Phase 4: Computing rank changes ==="

for dataset in dl19 dl20; do
    echo "  -> $dataset"
    scored_path="$SCORED_DIR/$dataset"
    output_path="$RANK_CHANGES_DIR/$dataset"

    for run_file in "$scored_path"/*.tsv; do
        fname=$(basename "$run_file")

        # Skip the baseline file: normal_t5.small.tsv has no 'augmented_score'
        # column and is not an attack result — it is the reference, not an input.
        if [[ "$fname" == normal_* ]]; then
            continue
        fi

        python "$EVALUATE_SCRIPT" \
            --run_file "$run_file" \
            --normal_dir "$scored_path" \
            --res_dump "$output_path"
    done

    FILE_COUNT=$(ls "$output_path/" | wc -l | tr -d ' ')
    echo "     $FILE_COUNT rank change files written to $output_path/"
done

# =============================================================================
# PHASE 5 — Aggregate rank changes and generate t5.small-only LaTeX table
#
# aggregate_rank_changes.py reads all *_rank_changes.tsv.gz files, picks
# the best (position, n_tok) per (token, model) by mean MRC across datasets,
# runs a Bonferroni-corrected Wilcoxon significance test, and writes a
# long-format TSV: token, model, dataset, metric, value, position, n_tok, sig
#
# make_scale_table_t5small.py is a t5.small-only variant of the paper's
# make_scale_table.py. It produces a 3-column LaTeX table (token, DL19, DL20)
# so you can visually verify MRC and Success Rate against Table 3 in the paper.
# =============================================================================

AGGREGATE_SCRIPT="$(dirname "${BASH_SOURCE[0]}")/aggregate_rank_changes.py"
TABLE_SCRIPT="$(dirname "${BASH_SOURCE[0]}")/make_scale_table_t5small.py"
AGGREGATED_TSV="$(dirname "${BASH_SOURCE[0]}")/aggregated_t5small.tsv"
TABLE_TEX="$(dirname "${BASH_SOURCE[0]}")/table3_t5small.tex"

echo ""
echo "=== Phase 5: Aggregating rank changes ==="
python "$AGGREGATE_SCRIPT" \
    --rank_changes_dir "$RANK_CHANGES_DIR" \
    --out_file "$AGGREGATED_TSV"
echo "  Written: $AGGREGATED_TSV"

echo ""
echo "=== Phase 5: Generating t5.small LaTeX table ==="
python "$TABLE_SCRIPT" \
    --run_file "$AGGREGATED_TSV" \
    --out_file "$TABLE_TEX"
echo "  Written: $TABLE_TEX"

# =============================================================================
# PHASE 6 — Clean up large intermediate files to recover quota
#
# Keeps:
#   runs/rank_changes/   — the final per-document rank change files (~1-2 GB)
#   aggregated_t5small.tsv
#   table3_t5small.tex
#
# Deletes:
#   runs/injected/       — attacked TSV files, ~15 GB (375 files × ~40 MB each)
#   runs/scored/         — scored TSV files,   ~15 GB (375 files × ~40 MB each)
#
# Note: rank_changes files are kept so you can re-run Phase 5 without re-scoring.
#       If you also want to free that space: rm -rf "$RANK_CHANGES_DIR"
# =============================================================================

echo ""
echo "=== Phase 6: Cleaning up intermediate files to recover disk quota ==="

echo "  Space used before cleanup:"
du -sh "$INJECTED_DIR" "$SCORED_DIR" 2>/dev/null || true

rm -rf "$INJECTED_DIR"
rm -rf "$SCORED_DIR"

echo "  Deleted: $INJECTED_DIR"
echo "  Deleted: $SCORED_DIR"
echo "  Kept:    $RANK_CHANGES_DIR"
echo "  Kept:    $AGGREGATED_TSV"
echo "  Kept:    $TABLE_TEX"

# =============================================================================
# SUMMARY
# =============================================================================

echo ""
echo "============================================================"
echo "  Done."
echo "============================================================"
echo ""
echo "  Rank change files (kept):"
echo "    DL19: $(ls "$RANK_CHANGES_DIR/dl19/" | wc -l | tr -d ' ') files in $RANK_CHANGES_DIR/dl19/"
echo "    DL20: $(ls "$RANK_CHANGES_DIR/dl20/" | wc -l | tr -d ' ') files in $RANK_CHANGES_DIR/dl20/"
echo ""
echo "  Aggregated results:  $AGGREGATED_TSV"
echo "  LaTeX table:         $TABLE_TEX"
echo ""
echo "  To verify against the paper, compare table3_t5small.tex with"
echo "  Table 3 (monoT5-small columns) in https://arxiv.org/pdf/2403.07654"
echo ""
echo "  To also reproduce monoT5-base results, run reproduce_t5_base.sh"
echo "  (same pipeline but --name t5.base, --model_name_or_path castorini/monot5-base-msmarco)"
