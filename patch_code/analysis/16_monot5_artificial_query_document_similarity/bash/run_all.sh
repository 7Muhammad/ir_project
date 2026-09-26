#!/usr/bin/env bash
# bash/run_all.sh — Experiment 16 full pipeline (artificial query-document similarity).
#
# Usage:
#   bash bash/run_all.sh [CONFIG] [--force] [--attack-start I] [--attack-end J]
#
#   --force                recompute every stage (also rebuilds the manifests)
#   --attack-start/-end    restrict stage 03 to a deterministic slice of the attack
#                          grid (to split across SLURM jobs); stages 04-05 then run
#                          only once all attacks are complete.
#
# SLURM (from /home/ghoummaid/IR):
#   bash run_job.sh --job-name monot5_exp16_similarity --gpu-type L40 --gpu-count 1 \
#     --cores 8 --time 24:00:00 \
#     --output-dir ./patch_code/analysis/16_monot5_artificial_query_document_similarity/slurm_logs \
#     --command "cd /home/ghoummaid/IR/patch_code/analysis/16_monot5_artificial_query_document_similarity && bash bash/run_all.sh configs/default.yaml"

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

echo "=== Experiment 16: artificial query-document similarity ==="
echo "Config: $CONFIG"
"$PY" -c "import torch; print('torch', torch.__version__, 'cuda available:', torch.cuda.is_available())"

stage "00 prepare manifests";          "$PY" scripts/00_prepare_manifests.py --config "$CONFIG" $FORCE
stage "01 sanity + clean similarity";  "$PY" scripts/01_run_clean_similarity.py --config "$CONFIG" $FORCE
stage "02 qrel similarity";            "$PY" scripts/02_run_qrel_similarity.py --config "$CONFIG" $FORCE
stage "03 attack/control similarity";  "$PY" scripts/03_run_attack_similarity.py --config "$CONFIG" $FORCE "${SLICE[@]}"
stage "07 decoder message similarity"; "$PY" scripts/07_run_decoder_message_similarity.py --config "$CONFIG" $FORCE "${SLICE[@]}"
stage "09 head-level similarity";      "$PY" scripts/09_run_head_similarity.py --config "$CONFIG" $FORCE "${SLICE[@]}"
stage "11 decoder probe similarity";   "$PY" scripts/11_run_decoder_probe_similarity.py --config "$CONFIG" $FORCE "${SLICE[@]}"
stage "13 all encoder heads L9-L11";  "$PY" scripts/13_run_late_encoder_heads.py --config "$CONFIG" $FORCE "${SLICE[@]}"
if [[ ${#SLICE[@]} -gt 0 ]]; then
    echo "=== attack slice given: skipping stages 04-14 (run without a slice once all slices finish) ==="
    exit 0
fi
stage "04 analyze";                    "$PY" scripts/04_analyze.py --config "$CONFIG" $FORCE
stage "05 plots";                      "$PY" scripts/05_plot.py --config "$CONFIG" $FORCE
stage "06 encoder level plot (4 populations)"; "$PY" scripts/06_plot_successful_vs_genuine_similarity.py --config "$CONFIG"
stage "08 decoder level plot (4 populations)"; "$PY" scripts/08_plot_decoder_message_similarity.py --config "$CONFIG"
stage "10 head-level tables + plots";  "$PY" scripts/10_plot_head_similarity.py --config "$CONFIG"
stage "12 decoder probe plot (4 populations)"; "$PY" scripts/12_plot_decoder_probe_similarity.py --config "$CONFIG"
stage "14 all encoder heads L9-L11 plot"; "$PY" scripts/14_plot_late_encoder_heads.py --config "$CONFIG"
echo ""
echo "=== [$(date '+%F %T')] Done. Outputs under the config's outputs.base_dir ==="
