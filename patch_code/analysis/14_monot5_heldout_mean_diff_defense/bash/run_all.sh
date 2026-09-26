#!/usr/bin/env bash
# bash/run_all.sh — Experiment 14 full pipeline: derive candidate heads ->
# prepare pair-disjoint splits -> cache baselines -> fit train-only
# directions -> validation scale sweep -> select scales -> held-out IID
# test + clean damage -> attack-OOD (leave-token/position/repetition-out)
# -> aggregate -> plots.
#
# Usage (interactive):
#   conda activate advseq2seq
#   bash bash/run_all.sh [config_path] [--force] [--skip-ood]
#
#   --skip-ood skips script 07 (attack-OOD, ~85% of total compute — see
#   DECISIONS.md item 16). Scripts 08/09 degrade gracefully with no OOD
#   data present (IID-only summary rows; Figure C's panels show "no data").
#   Re-run without --skip-ood later to fill OOD in (each fold is its own
#   resume-safe unit, so this does not repeat the IID stages).
#
# Usage (SLURM via run_job.sh from /home/ghoummaid/IR):
#   bash run_job.sh \
#     --job-name monot5_exp14_heldout_defense \
#     --gpu-type L40 --gpu-count 1 --cores 8 \
#     --time 48:00:00 --output-dir ./slurm_logs \
#     --command "cd /home/ghoummaid/IR/patch_code/analysis/14_monot5_heldout_mean_diff_defense && \
#                bash bash/run_all.sh configs/default.yaml --skip-ood"

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="$(dirname "$SCRIPT_DIR")"

CONFIG="${EXP_DIR}/configs/default.yaml"
FORCE_FLAG=""
SKIP_OOD=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --force)    FORCE_FLAG="--force"; shift ;;
        --skip-ood) SKIP_OOD="1"; shift ;;
        --*)        echo "Unknown option: $1"; exit 1 ;;
        *)          CONFIG="$1"; shift ;;
    esac
done

CONFIG="$(realpath "$CONFIG")"
echo "=== Experiment 14: held-out single-head mean-diff defense ==="
echo "Config: $CONFIG"

CANDIDATE_HEADS_OUT="$(python3 -c "
import yaml, pathlib
cfg = yaml.safe_load(open('$CONFIG'))
base = (pathlib.Path('$CONFIG').parent / cfg['outputs']['base_dir']).resolve()
print(base / 'candidate_heads.json')
")"

python3 "${EXP_DIR}/scripts/00_derive_candidate_heads.py" --out "$CANDIDATE_HEADS_OUT"
python3 "${EXP_DIR}/scripts/01_prepare_splits.py" --config "$CONFIG"   # cheap + deterministic, no --force needed
python3 "${EXP_DIR}/scripts/02_cache_baselines.py" --config "$CONFIG" $FORCE_FLAG
python3 "${EXP_DIR}/scripts/03_fit_directions.py" --config "$CONFIG" $FORCE_FLAG
python3 "${EXP_DIR}/scripts/04_run_validation.py" --config "$CONFIG" $FORCE_FLAG
python3 "${EXP_DIR}/scripts/05_select_scales.py" --config "$CONFIG" $FORCE_FLAG
python3 "${EXP_DIR}/scripts/06_run_iid_test.py" --config "$CONFIG" $FORCE_FLAG
if [[ -z "$SKIP_OOD" ]]; then
    python3 "${EXP_DIR}/scripts/07_run_attack_ood.py" --config "$CONFIG" $FORCE_FLAG
else
    echo "=== --skip-ood: skipping script 07 (attack-OOD) ==="
fi
python3 "${EXP_DIR}/scripts/08_aggregate.py" --config "$CONFIG"
python3 "${EXP_DIR}/scripts/09_make_plots.py" --config "$CONFIG"

echo "=== Done. See outputs under the config's outputs.base_dir ==="
