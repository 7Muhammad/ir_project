#!/usr/bin/env bash
# bash/run_encoder_head_patching.sh — Experiment 11 full pipeline:
# canonical head patch (Step 1) -> full 105-attack sweep + additivity
# (Steps 2-3) -> layer 8-11 attention crossref (Step 4) -> attack-token
# hub analysis (Step 5) -> aggregate/plots/summary.
#
# Usage (interactive):
#   conda activate advseq2seq
#   bash bash/run_encoder_head_patching.sh [config_path] [--force]
#
# Usage (SLURM via run_job.sh from /home/ghoummaid/IR):
#   bash run_job.sh \
#     --job-name monot5_encoder_head_patching \
#     --gpu-type L40 --gpu-count 1 --cores 8 \
#     --time 12:00:00 --output-dir ./slurm_logs \
#     --command "cd /home/ghoummaid/IR/patch_code/analysis/11_monot5_encoder_head_patching && \
#                bash bash/run_encoder_head_patching.sh configs/default.yaml"

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
echo "=== Experiment 11: encoder per-head causal patching ==="
echo "Config: $CONFIG"

python3 "${EXP_DIR}/scripts/01_encoder_head_patch_canonical.py" --config "$CONFIG" $FORCE_FLAG
python3 "${EXP_DIR}/scripts/02_encoder_head_patch_full_sweep.py" --config "$CONFIG" $FORCE_FLAG
python3 "${EXP_DIR}/scripts/03_layer9_head_attention_crossref.py" --config "$CONFIG"
python3 "${EXP_DIR}/scripts/04_attack_token_hub_analysis.py" --config "$CONFIG"
python3 "${EXP_DIR}/scripts/05_aggregate_and_plot.py" --config "$CONFIG"

echo "=== Done. Outputs under $(dirname "$EXP_DIR")/11_monot5_encoder_head_patching/outputs ==="
