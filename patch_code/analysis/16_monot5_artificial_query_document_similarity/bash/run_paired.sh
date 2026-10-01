#!/usr/bin/env bash
# bash/run_paired.sh — Experiment 16 PAIRED extension (stages 17-21):
# padded control -> successful attack on every judged attackable DL19 pair x 105 attacks.
#
# Usage:
#   bash bash/run_paired.sh [CONFIG] [--force] [--attack-start I] [--attack-end J]
#
#   --force                recompute every paired stage (also rebuilds the paired manifest)
#   --attack-start/-end    restrict stage 18 to a slice of the attack grid (split across jobs);
#                          stages 19-21 then run only once every attack is complete.
#
# SLURM (from /home/ghoummaid/IR):
#   bash run_job.sh --job-name monot5_exp16_paired --gpu-type L40 --gpu-count 1 --cores 8 --time 24:00:00 \
#     --output-dir ./patch_code/analysis/16_monot5_artificial_query_document_similarity/slurm_logs \
#     --command "cd /home/ghoummaid/IR/patch_code/analysis/16_monot5_artificial_query_document_similarity && bash bash/run_paired.sh configs/default.yaml"

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

echo "=== Experiment 16 (paired): padded control -> successful attack ==="
echo "Config: $CONFIG"
"$PY" -c "import torch; print('torch', torch.__version__, 'cuda available:', torch.cuda.is_available())"

stage "17 paired manifest";                 "$PY" scripts/17_prepare_paired_manifest.py --config "$CONFIG" $FORCE
stage "18 paired control/attack forward";   "$PY" scripts/18_run_paired_forward.py --config "$CONFIG" $FORCE "${SLICE[@]}"
if [[ ${#SLICE[@]} -gt 0 ]]; then
    echo "=== attack slice given: skipping stages 19-21 (run without a slice once all slices finish) ==="
    exit 0
fi
stage "19 paired abnormality";              "$PY" scripts/19_paired_anomaly.py --config "$CONFIG"
stage "20 paired top-k detector";           "$PY" scripts/20_paired_detector.py --config "$CONFIG"
stage "21 paired plots + final summary";    "$PY" scripts/21_plot_paired.py --config "$CONFIG"
echo ""
echo "=== [$(date '+%F %T')] Done. See <outputs>/21_paired_report/final_summary.json ==="
