#!/usr/bin/env bash
# bash/run_multi_attack_grid.sh
# ==============================
# Full grid run: multi-attack pipeline → cross-attack comparison → git push.
#
# Resume-safe: script 10 checks status.json + output files before re-running
# any attack.  Attacks that already have all outputs (plots, CSVs) are skipped
# automatically, so you can safely re-submit this script after a failure or
# timeout without re-running completed work.
#
# Usage (interactive):
#   conda activate advseq2seq
#   bash bash/run_multi_attack_grid.sh [config_path] [--no-push]
#
# Usage (SLURM via run_job.sh, from the repo root /home/ghoummaid/IR):
#   bash run_job.sh \
#     --job-name monot5_multi_attack_grid \
#     --gpu-type L40 --gpu-count 1 --cores 8 \
#     --time 48:00:00 --output-dir ./slurm_logs \
#     --command "cd /home/ghoummaid/IR/patch_code/analysis/monot5_layer_patching && \
#                bash bash/run_multi_attack_grid.sh configs/multi_attack.yaml"
#
# Arguments:
#   [config_path]   Path to YAML config (default: configs/multi_attack.yaml)
#   --no-push       Skip the final git push step

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
REPO_ROOT="$(dirname "$(dirname "$(dirname "$PROJECT_ROOT")")")"   # /home/ghoummaid/IR

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
CONFIG="${PROJECT_ROOT}/configs/multi_attack.yaml"
DO_PUSH=true

for arg in "$@"; do
    case "$arg" in
        --no-push)
            DO_PUSH=false
            ;;
        --*)
            echo "ERROR: Unknown option: $arg"
            echo "Usage: bash run_multi_attack_grid.sh [config_path] [--no-push]"
            exit 1
            ;;
        *)
            CONFIG="$arg"
            ;;
    esac
done

if [[ ! -f "$CONFIG" ]]; then
    echo "ERROR: Config file not found: $CONFIG"
    exit 1
fi

CONFIG="$(realpath "$CONFIG")"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
hr() { printf '%0.s=' {1..60}; echo; }
timestamp() { date '+%Y-%m-%d %H:%M:%S'; }

# ---------------------------------------------------------------------------
# Stage 1 — Multi-attack pipeline (resume-safe)
# ---------------------------------------------------------------------------
hr
echo "  monoT5 Multi-Attack Grid Run"
echo "  $(timestamp)"
echo "  Config : $CONFIG"
echo "  Repo   : $REPO_ROOT"
hr
echo ""

echo ">>> [$(timestamp)] Stage 1: multi-attack pipeline (script 10) ..."
python3 "${PROJECT_ROOT}/scripts/10_run_multi_attack_pipeline.py" --config "$CONFIG"
PIPELINE_EXIT=$?

if [[ $PIPELINE_EXIT -ne 0 ]]; then
    echo ""
    echo "ERROR: script 10 exited with code $PIPELINE_EXIT."
    echo "Some attacks may have failed.  Check status.json files in outputs/attacks/."
    echo "Re-running this script will resume from where it left off."
    exit $PIPELINE_EXIT
fi

echo ""
echo ">>> [$(timestamp)] Stage 1 complete."

# ---------------------------------------------------------------------------
# Stage 2 — Cross-attack comparison
# ---------------------------------------------------------------------------
echo ""
echo ">>> [$(timestamp)] Stage 2: cross-attack comparison (script 11) ..."
python3 "${PROJECT_ROOT}/scripts/11_compare_attacks.py" --config "$CONFIG"
COMPARE_EXIT=$?

if [[ $COMPARE_EXIT -ne 0 ]]; then
    echo ""
    echo "ERROR: script 11 exited with code $COMPARE_EXIT."
    exit $COMPARE_EXIT
fi

echo ""
echo ">>> [$(timestamp)] Stage 2 complete."

# ---------------------------------------------------------------------------
# Stage 3 — Git commit + push
# ---------------------------------------------------------------------------
if [[ "$DO_PUSH" == "false" ]]; then
    echo ""
    echo ">>> [$(timestamp)] Skipping git push (--no-push specified)."
    hr
    echo "  All stages complete."
    hr
    exit 0
fi

echo ""
echo ">>> [$(timestamp)] Stage 3: committing and pushing results ..."

cd "$REPO_ROOT"

# Stage all new/modified output files (lightweight: CSVs, JSONs, PNGs, status files).
# Activation .pt files are excluded by .gitignore (or are already cleaned up).
git add \
    "patch_code/analysis/monot5_layer_patching/outputs/" \
    "patch_code/analysis/monot5_layer_patching/configs/multi_attack.yaml" \
    "patch_code/analysis/monot5_layer_patching/scripts/10_run_multi_attack_pipeline.py" \
    "patch_code/analysis/monot5_layer_patching/bash/run_multi_attack_grid.sh" \
    2>/dev/null || true

# Only commit if there is anything staged.
if git diff --cached --quiet; then
    echo "  Nothing new to commit — working tree already up to date."
else
    COMMIT_MSG="results: multi-attack grid (105 attacks) — $(timestamp)"
    git commit -m "$COMMIT_MSG"
    echo "  Committed: $COMMIT_MSG"
fi

git push origin HEAD
GIT_EXIT=$?

if [[ $GIT_EXIT -ne 0 ]]; then
    echo "ERROR: git push failed (exit $GIT_EXIT)."
    exit $GIT_EXIT
fi

echo "  Pushed to origin."

hr
echo "  All stages complete.  $(timestamp)"
hr
