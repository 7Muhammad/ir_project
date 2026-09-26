#!/usr/bin/env bash
# bash/run_all.sh — Experiment 15 full pipeline (attack-token -> query edge patching).
#
# Usage:
#   bash bash/run_all.sh [CONFIG] [--force] [--attack-start I] [--attack-end J]
#
#   --force                recompute every stage (also rewrites the manifest!)
#   --attack-start/-end    restrict the model stages (02-04) to a deterministic
#                          slice of the grid (for splitting across SLURM jobs);
#                          stages 05-07 run only once all attacks are complete.
#
# SLURM (from /home/ghoummaid/IR):
#   bash run_job.sh --job-name monot5_exp15_query_flow --gpu-type L40 --gpu-count 1 \
#     --cores 8 --time 24:00:00 \
#     --output-dir ./patch_code/analysis/15_monot5_attack_to_query_flow/slurm_logs \
#     --command "cd /home/ghoummaid/IR/patch_code/analysis/15_monot5_attack_to_query_flow && bash bash/run_all.sh configs/default.yaml"

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="$(dirname "$SCRIPT_DIR")"
PY="/home/ghoummaid/miniconda3/envs/advseq2seq/bin/python"

CONFIG="${EXP_DIR}/configs/default.yaml"
FORCE=""
SLICE=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --force) FORCE="--force"; shift ;;
        --attack-start) SLICE+=(--attack-start "$2"); shift 2 ;;
        --attack-end) SLICE+=(--attack-end "$2"); shift 2 ;;
        --*) echo "Unknown option: $1"; exit 1 ;;
        *) CONFIG="$1"; shift ;;
    esac
done
CONFIG="$(realpath "$CONFIG")"
cd "$EXP_DIR"

stage() { echo ""; echo "=== [$(date '+%F %T')] $1 ==="; }

echo "=== Experiment 15: attack-token -> query information flow ==="
echo "Config: $CONFIG"
"$PY" -c "import torch; print('torch', torch.__version__, 'cuda available:', torch.cuda.is_available())"

stage "00 prepare samples";            "$PY" scripts/00_prepare_samples.py --config "$CONFIG" $FORCE
stage "01 sanity checks";              "$PY" scripts/01_sanity_checks.py --config "$CONFIG" $FORCE
stage "02 all-query edges";            "$PY" scripts/02_run_all_query_edges.py --config "$CONFIG" $FORCE "${SLICE[@]}"
stage "04 whole-head reference";       "$PY" scripts/04_run_whole_head_reference.py --config "$CONFIG" $FORCE "${SLICE[@]}"
stage "03 single-query-token edges";   "$PY" scripts/03_run_single_query_edges.py --config "$CONFIG" $FORCE "${SLICE[@]}"
if [[ ${#SLICE[@]} -gt 0 ]]; then
    echo "=== attack slice given: skipping stages 05-07 (run without a slice once all slices finish) ==="
    exit 0
fi
stage "05 aggregate";                  "$PY" scripts/05_aggregate.py --config "$CONFIG" $FORCE
stage "06 statistics";                 "$PY" scripts/06_statistics.py --config "$CONFIG" $FORCE
stage "07 plots";                      "$PY" scripts/07_plot.py --config "$CONFIG" $FORCE
stage "08 post-hoc robustness (descriptive)"; "$PY" scripts/08_posthoc_robustness.py --config "$CONFIG" $FORCE
echo ""
echo "=== [$(date '+%F %T')] Done. Outputs under the config's outputs.base_dir ==="
