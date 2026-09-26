#!/usr/bin/env bash
# bash/run_sweep.sh — Experiment 13: main 558-path sweep + aggregation +
# plots. Safe to re-run: scripts/02_run_sweep.py checkpoints per
# (attack, example) and skips already-completed work.
#
# Usage (interactive):
#   conda activate advseq2seq
#   bash bash/run_sweep.sh [config_path] [--max-examples N]
#
# Usage (SLURM via run_job.sh from /home/ghoummaid/IR):
#   bash run_job.sh \
#     --job-name monot5_exp13_path_patching \
#     --gpu-type L40 --gpu-count 1 --cores 8 \
#     --time 24:00:00 --output-dir ./slurm_logs \
#     --command "cd /home/ghoummaid/IR/patch_code/analysis/13_monot5_encoder_decoder_path_patching && \
#                bash bash/run_sweep.sh configs/default.yaml"

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="$(dirname "$SCRIPT_DIR")"

CONFIG="${EXP_DIR}/configs/default.yaml"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --max-examples) EXTRA_ARGS+=("--max-examples" "$2"); shift 2 ;;
        --*)            EXTRA_ARGS+=("$1"); shift ;;
        *)              CONFIG="$1"; shift ;;
    esac
done

CONFIG="$(realpath "$CONFIG")"
echo "=== Experiment 13: encoder-decoder head path patching ==="
echo "Config: $CONFIG"

python3 "${EXP_DIR}/scripts/02_run_sweep.py" --config "$CONFIG" "${EXTRA_ARGS[@]}"
python3 "${EXP_DIR}/scripts/03_aggregate.py" --config "$CONFIG"
python3 "${EXP_DIR}/scripts/04_make_plots.py" --config "$CONFIG"

echo "=== Done. Outputs under ${EXP_DIR}/outputs ==="
