#!/usr/bin/env bash
# bash/run_all.sh — Experiment 12 full pipeline: causal query-word/structural
# patching (105 attacks x causal n) -> attention analysis (105 attacks x
# attention n) -> aggregate -> plots -> diagnostics.
#
# Usage (interactive):
#   conda activate advseq2seq
#   bash bash/run_all.sh [config_path] [--force]
#
# Usage (SLURM via run_job.sh from /home/ghoummaid/IR):
#   bash run_job.sh \
#     --job-name monot5_query_token_localization \
#     --gpu-type L40 --gpu-count 1 --cores 8 \
#     --time 24:00:00 --output-dir ./slurm_logs \
#     --command "cd /home/ghoummaid/IR/patch_code/analysis/12_query_token_localization && \
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
echo "=== Experiment 12: query-token localization ==="
echo "Config: $CONFIG"

python3 "${EXP_DIR}/scripts/01_run_causal.py" --config "$CONFIG" $FORCE_FLAG
python3 "${EXP_DIR}/scripts/02_run_attention.py" --config "$CONFIG" $FORCE_FLAG
python3 "${EXP_DIR}/scripts/03_aggregate.py" --config "$CONFIG"
python3 "${EXP_DIR}/scripts/04_make_plots.py" --config "$CONFIG"
python3 "${EXP_DIR}/scripts/05_diagnostics.py" --config "$CONFIG"

echo "=== Done. Outputs under $(dirname "$EXP_DIR")/12_query_token_localization/outputs ==="
