#!/usr/bin/env bash
# bash/run_all.sh [config] [--force] — stage 01 (forward + attention statistics + validation), stage 02 (analysis + plots).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="$(dirname "$SCRIPT_DIR")"
PY=/home/ghoummaid/miniconda3/envs/advseq2seq/bin/python
CFG="${1:-${EXP_DIR}/configs/default.yaml}"
shift || true
cd "$EXP_DIR"
echo "=== Exp 17 stage 01: attention statistics ($CFG) ==="
$PY scripts/01_attention_stats.py --config "$CFG" "$@"
echo "=== Exp 17 stage 02: analysis ==="
$PY scripts/02_analyze.py --config "$CFG" "$@"
