#!/usr/bin/env bash
# bash/run_template_tokens.sh — Experiment 6 extension full pipeline:
# tokenizer ground truth -> causal patch (canonical) -> attention mass ->
# full 105-attack sweep -> aggregate/additivity-check/plots/summary.
#
# Usage (interactive):
#   conda activate advseq2seq
#   bash bash/run_template_tokens.sh [config_path] [--force]
#
# Usage (SLURM via run_job.sh from /home/ghoummaid/IR):
#   bash run_job.sh \
#     --job-name monot5_template_tokens \
#     --gpu-type L40 --gpu-count 1 --cores 8 \
#     --time 24:00:00 --output-dir ./slurm_logs \
#     --command "cd /home/ghoummaid/IR/patch_code/analysis/06_monot5_query_doc_patching && \
#                bash bash/run_template_tokens.sh configs/template_tokens.yaml"

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="$(dirname "$SCRIPT_DIR")"

CONFIG="${EXP_DIR}/configs/template_tokens.yaml"
FORCE_FLAG=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --force) FORCE_FLAG="--force"; shift ;;
        --*)     echo "Unknown option: $1"; exit 1 ;;
        *)       CONFIG="$1"; shift ;;
    esac
done

CONFIG="$(realpath "$CONFIG")"
echo "=== Experiment 6 extension: template-token sink/signal analysis ==="
echo "Config: $CONFIG"

python3 "${EXP_DIR}/scripts/05_template_tokenizer_ground_truth.py" --config "$CONFIG"
python3 "${EXP_DIR}/scripts/06_template_causal_patch.py" --config "$CONFIG" $FORCE_FLAG
python3 "${EXP_DIR}/scripts/07_template_attention_mass.py" --config "$CONFIG"
python3 "${EXP_DIR}/scripts/08_template_full_sweep.py" --config "$CONFIG" $FORCE_FLAG
python3 "${EXP_DIR}/scripts/09_template_aggregate_and_plot.py" --config "$CONFIG"

echo "=== Done. Outputs under $(dirname "$EXP_DIR")/06_monot5_query_doc_patching/outputs/template_tokens ==="
