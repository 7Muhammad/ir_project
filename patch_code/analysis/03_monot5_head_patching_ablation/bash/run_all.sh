#!/usr/bin/env bash
# bash/run_all.sh — Experiment 3 full pipeline: 4 head-grid runs → aggregate → plots.
#
# Resume-safe: script 01 skips (run, attack) units whose status.json says
# success, so re-submitting after a failure/timeout continues where it stopped.
#
# Usage (interactive):
#   conda activate advseq2seq
#   bash bash/run_all.sh [config_path] [--run grid_a|grid_b|clean_a|clean_b|all] [--force]
#
# Usage (SLURM via run_job.sh from /home/ghoummaid/IR):
#   bash run_job.sh \
#     --job-name monot5_head_patching \
#     --gpu-type L40 --gpu-count 1 --cores 8 \
#     --time 48:00:00 --output-dir ./slurm_logs \
#     --command "cd /home/ghoummaid/IR/patch_code/analysis/03_monot5_head_patching_ablation && \
#                bash bash/run_all.sh configs/default.yaml"

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="$(dirname "$SCRIPT_DIR")"

CONFIG="${EXP_DIR}/configs/default.yaml"
RUN="all"
FORCE_FLAG=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --run)   RUN="$2"; shift 2 ;;
        --force) FORCE_FLAG="--force"; shift ;;
        --*)     echo "Unknown option: $1"; exit 1 ;;
        *)       CONFIG="$1"; shift ;;
    esac
done

CONFIG="$(realpath "$CONFIG")"
echo "=== Experiment 3: per-head patching + ablation ==="
echo "Config: $CONFIG   Run: $RUN"

python3 "${EXP_DIR}/scripts/01_run_head_grid.py" --config "$CONFIG" --run "$RUN" $FORCE_FLAG
python3 "${EXP_DIR}/scripts/02_aggregate.py"     --config "$CONFIG"
python3 "${EXP_DIR}/scripts/03_make_plots.py"    --config "$CONFIG"

echo "=== Done. Outputs under $(dirname "$EXP_DIR")/$(basename "$EXP_DIR")/outputs ==="
