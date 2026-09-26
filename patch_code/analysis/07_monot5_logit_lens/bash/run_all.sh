#!/usr/bin/env bash
# bash/run_all.sh — Experiment 7 full pipeline: grid+canonical -> aggregate -> plots.
#
# DO NOT RUN until scripts/01_run_pilot.py's extrapolated runtime has been
# reviewed and configs/default.yaml's runs.grid.n_examples is confirmed.
#
# Usage (interactive):
#   conda activate advseq2seq
#   bash bash/run_all.sh [config_path] [--run grid|canonical|all] [--force]
#
# Usage (SLURM via run_job.sh from /home/ghoummaid/IR):
#   bash run_job.sh \
#     --job-name monot5_logit_lens \
#     --gpu-type L40 --gpu-count 1 --cores 8 \
#     --time 12:00:00 --output-dir ./slurm_logs \
#     --command "cd /home/ghoummaid/IR/patch_code/analysis/07_monot5_logit_lens && \
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
echo "=== Experiment 7: logit lens on monoT5 decoder cross-attention ==="
echo "Config: $CONFIG   Run: $RUN"

python3 "${EXP_DIR}/scripts/02_run_grid.py" --config "$CONFIG" --run "$RUN" $FORCE_FLAG
python3 "${EXP_DIR}/scripts/03_aggregate.py" --config "$CONFIG"
python3 "${EXP_DIR}/scripts/04_make_plots.py" --config "$CONFIG"

echo "=== Done. Outputs under $(dirname "$EXP_DIR")/07_monot5_logit_lens/outputs ==="
