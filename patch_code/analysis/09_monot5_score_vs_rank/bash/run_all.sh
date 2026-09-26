#!/usr/bin/env bash
# bash/run_all.sh — Score-vs-Rank full pipeline: candidate scores -> per-example
# score/rank -> aggregate -> plot.
#
# Cheap by design: only script 01 touches the model (~4,200 candidate
# forward passes, one-time); scripts 02-04 are pure CSV/dict logic. No SLURM
# needed, though it's fine to run through run_job.sh too.
#
# Usage (interactive):
#   conda activate advseq2seq
#   bash bash/run_all.sh [config_path]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="$(dirname "$SCRIPT_DIR")"

CONFIG="${EXP_DIR}/configs/default.yaml"
if [[ $# -gt 0 ]]; then
    CONFIG="$1"
fi
CONFIG="$(realpath "$CONFIG")"

echo "=== Score-vs-Rank Comparison Experiment ==="
echo "Config: $CONFIG"

python3 "${EXP_DIR}/scripts/01_build_candidate_scores.py" --config "$CONFIG"
python3 "${EXP_DIR}/scripts/02_compute_score_rank.py"      --config "$CONFIG"
python3 "${EXP_DIR}/scripts/03_aggregate.py"                --config "$CONFIG"
python3 "${EXP_DIR}/scripts/04_make_plot.py"                --config "$CONFIG"

echo "=== Done. Outputs under $(dirname "$EXP_DIR")/$(basename "$EXP_DIR")/outputs ==="
