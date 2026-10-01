#!/usr/bin/env bash
# bash/run_tests.sh — pytest suite (CPU; real monoT5 tokenizer, synthetic tensors, Exp 16 manifest if present).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(dirname "$SCRIPT_DIR")"
CUDA_VISIBLE_DEVICES="" /home/ghoummaid/miniconda3/envs/advseq2seq/bin/python -m pytest tests/ -v "$@"
