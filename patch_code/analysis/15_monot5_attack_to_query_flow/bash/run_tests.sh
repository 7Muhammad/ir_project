#!/usr/bin/env bash
# bash/run_tests.sh — pytest suite (CPU; tiny random T5 + real tokenizer / Exp 01 pools).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(dirname "$SCRIPT_DIR")"
CUDA_VISIBLE_DEVICES="" /home/ghoummaid/miniconda3/envs/advseq2seq/bin/python -m pytest tests/ -v "$@"
