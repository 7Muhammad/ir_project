#!/usr/bin/env bash
# bash/run_all.sh [config] [--force] — stage 01 (forward + on-the-fly metrics + CKA statistics), stage 02 (analysis + plots).
# Needs Exp 16 stages 17 (paired manifest) and 18 (paired forward scores) complete.
#
# SLURM (from /home/ghoummaid/IR):
#   bash run_job.sh --job-name monot5_exp18_repr --gpu-type L40 --gpu-count 1 --cores 8 --time 08:00:00 \
#     --output-dir ./patch_code/analysis/18_monot5_encoder_representation_change/slurm_logs \
#     --command "cd /home/ghoummaid/IR/patch_code/analysis/18_monot5_encoder_representation_change && bash bash/run_all.sh"
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="$(dirname "$SCRIPT_DIR")"
PY=/home/ghoummaid/miniconda3/envs/advseq2seq/bin/python
CFG="${1:-${EXP_DIR}/configs/default.yaml}"
shift || true
cd "$EXP_DIR"
"$PY" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
echo "=== [$(date '+%F %T')] Exp 18 stage 01: representation-change forward ($CFG) ==="
"$PY" scripts/01_run_repr_change.py --config "$CFG" "$@"
echo "=== [$(date '+%F %T')] Exp 18 stage 02: analysis + figures ==="
"$PY" scripts/02_analyze_repr_change.py --config "$CFG" "$@"
echo "=== [$(date '+%F %T')] done ==="
