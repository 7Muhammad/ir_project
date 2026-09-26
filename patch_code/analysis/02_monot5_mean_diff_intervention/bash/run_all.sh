#!/usr/bin/env bash
# bash/run_all.sh — Experiment 2 full pipeline: flagged heads -> directions ->
# interventions -> aggregate -> plots.
#
# Resume-safe: script 01 skips tiers, and script 02 skips (tier, attack)
# units, whose status.json says success, so re-submitting after a
# failure/timeout continues where it stopped.
#
# Usage (interactive):
#   conda activate advseq2seq
#   bash bash/run_all.sh [config_path] [--force]
#
# Usage (SLURM via run_job.sh from /home/ghoummaid/IR):
#   bash run_job.sh \
#     --job-name monot5_mean_diff \
#     --gpu-type L40 --gpu-count 1 --cores 8 \
#     --time 48:00:00 --output-dir ./slurm_logs \
#     --command "cd /home/ghoummaid/IR/patch_code/analysis/02_monot5_mean_diff_intervention && \
#                bash bash/run_all.sh configs/default.yaml"

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="$(dirname "$SCRIPT_DIR")"

CONFIG="${EXP_DIR}/configs/default.yaml"
FORCE_FLAG=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --force) FORCE_FLAG="--force"; shift ;;
        --*)     echo "Unknown option: $1"; exit 1 ;;
        *)       CONFIG="$1"; shift ;;
    esac
done

CONFIG="$(realpath "$CONFIG")"
echo "=== Experiment 2: mean-diff direction analysis + intervention ==="
echo "Config: $CONFIG"

python3 "${EXP_DIR}/scripts/00_derive_flagged_heads.py"  --config "$CONFIG"
python3 "${EXP_DIR}/scripts/01_compute_directions.py"    --config "$CONFIG" $FORCE_FLAG
python3 "${EXP_DIR}/scripts/02_run_interventions.py"     --config "$CONFIG" $FORCE_FLAG
python3 "${EXP_DIR}/scripts/03_aggregate.py"              --config "$CONFIG"
python3 "${EXP_DIR}/scripts/04_make_plots.py"             --config "$CONFIG"

echo "=== Done. Outputs under $(dirname "$EXP_DIR")/$(basename "$EXP_DIR")/outputs ==="
