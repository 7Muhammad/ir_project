#!/bin/bash
# =============================================================================
# setup_env.sh
#
# Creates the conda environment required to run reproduce_t5_small.sh.
#
# Environment name: advseq2seq
# Python version:   3.10
#
# KEY DEPENDENCIES
# ----------------
#   openjdk 11     — Python-Terrier wraps Terrier (a Java IR system) via JPype.
#                    Java is not installed system-wide, so we install it through
#                    conda-forge. This avoids needing sudo.
#
#   torch (CUDA)   — Automatically detects the CUDA version available on this
#                    machine via `nvidia-smi` and installs the matching PyTorch
#                    CUDA build. Falls back to CPU if no GPU is found, but that
#                    will make Phase 3 (scoring) extremely slow (~19 days vs ~2h).
#
#   python-terrier  — PyTerrier framework (wraps Terrier for retrieval pipelines)
#   pyterrier-t5    — Provides MonoT5ReRanker used in scale_score.py
#   transformers    — HuggingFace model loading (monoT5 checkpoints from castorini/)
#   fire            — CLI argument parsing used by all repo scripts
#   pandas          — TSV file reading/writing throughout the pipeline
#   numpy           — Random position sampling in token_injection.py
#
# USAGE
#   bash setup_env.sh
#
# After setup:
#   conda activate advseq2seq
#   bash reproduce_t5_small.sh
# =============================================================================

set -euo pipefail

ENV_NAME="advseq2seq"
PYTHON_VERSION="3.10"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/ecir24-adversarial-evaluation" && pwd)"

echo "=== Creating conda environment: $ENV_NAME (Python $PYTHON_VERSION) ==="

# Remove existing environment if it exists, to ensure a clean install.
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "  Environment '$ENV_NAME' already exists — removing and recreating."
    conda env remove -n "$ENV_NAME" -y
fi

# Create environment with Python and Java together so conda can resolve
# compatibility between them in a single solve.
conda create -n "$ENV_NAME" \
    python="$PYTHON_VERSION" \
    "openjdk=11" \
    -c conda-forge \
    -y

# =============================================================================
# Detect CUDA version and install the matching PyTorch build.
#
# nvidia-smi reports the maximum CUDA version the driver supports.
# We map that to the closest PyTorch CUDA wheel that exists:
#   >= 12.1  → cu121  (covers 12.1, 12.2, 12.3, 12.4, ...)
#   >= 11.8  → cu118
#   >= 11.7  → cu117
#   anything else or no GPU → cpu (but warn loudly)
#
# If running inside a SLURM job on a GPU node, nvidia-smi will be available
# even if it isn't on the login node — the conda env is shared via NFS so
# it only needs to be created once (on the login node is fine), and the
# torch CUDA build will work on GPU nodes.
# =============================================================================

echo ""
echo "=== Detecting CUDA version ==="

TORCH_INDEX_URL="https://download.pytorch.org/whl/cpu"
CUDA_TAG="cpu"

if command -v nvidia-smi &>/dev/null; then
    # Extract "12.1" from "CUDA Version: 12.1"
    CUDA_VERSION=$(nvidia-smi | grep -oP "CUDA Version: \K[0-9]+\.[0-9]+" || true)
    if [[ -n "$CUDA_VERSION" ]]; then
        CUDA_MAJOR=$(echo "$CUDA_VERSION" | cut -d. -f1)
        CUDA_MINOR=$(echo "$CUDA_VERSION" | cut -d. -f2)
        echo "  Detected CUDA $CUDA_VERSION"

        if [[ "$CUDA_MAJOR" -gt 12 ]] || { [[ "$CUDA_MAJOR" -eq 12 ]] && [[ "$CUDA_MINOR" -ge 1 ]]; }; then
            TORCH_INDEX_URL="https://download.pytorch.org/whl/cu121"
            CUDA_TAG="cu121"
        elif [[ "$CUDA_MAJOR" -eq 11 ]] && [[ "$CUDA_MINOR" -ge 8 ]]; then
            TORCH_INDEX_URL="https://download.pytorch.org/whl/cu118"
            CUDA_TAG="cu118"
        elif [[ "$CUDA_MAJOR" -eq 11 ]] && [[ "$CUDA_MINOR" -ge 7 ]]; then
            TORCH_INDEX_URL="https://download.pytorch.org/whl/cu117"
            CUDA_TAG="cu117"
        else
            echo "  WARNING: CUDA $CUDA_VERSION is older than 11.7 — falling back to CPU build."
        fi
    else
        echo "  nvidia-smi found but could not parse CUDA version — falling back to CPU build."
    fi
else
    echo "  nvidia-smi not found on this node."
    echo ""
    echo "  *** WARNING: Installing CPU-only PyTorch. ***"
    echo "  *** Phase 3 (scoring) will take ~19 days instead of ~2 hours. ***"
    echo "  *** Run this script on a GPU node, or reinstall torch manually: ***"
    echo "  ***   conda run -n $ENV_NAME pip install torch --index-url https://download.pytorch.org/whl/cu121 ***"
    echo ""
fi

echo "  Installing torch with: --index-url $TORCH_INDEX_URL  ($CUDA_TAG)"
echo ""
echo "=== Installing PyTorch ($CUDA_TAG build) ==="
conda run -n "$ENV_NAME" pip install \
    torch \
    --index-url "$TORCH_INDEX_URL"

echo ""
echo "=== Installing core dependencies ==="
conda run -n "$ENV_NAME" pip install \
    transformers \
    pandas \
    numpy \
    tqdm \
    fire \
    sentencepiece

echo ""
echo "=== Installing PyTerrier and monoT5 plugin ==="
# python-terrier downloads Terrier JARs on first use (~100MB, cached in ~/.pyterrier).
# pyterrier-t5 provides MonoT5ReRanker used by scale_score.py.
conda run -n "$ENV_NAME" pip install \
    python-terrier \
    pyterrier-t5

echo ""
echo "=== Installing the repo package (advseq2seq) ==="
conda run -n "$ENV_NAME" pip install -e "$REPO_DIR"

echo ""
echo "=== Verifying installation ==="

conda run -n "$ENV_NAME" python - << 'EOF'
import sys
print(f"  Python:       {sys.version.split()[0]}")

import torch
cuda_ok = torch.cuda.is_available()
print(f"  torch:        {torch.__version__}  (CUDA available: {cuda_ok})")
if not cuda_ok:
    print("  *** WARNING: CUDA not available — scoring will be very slow on CPU ***")

import transformers
print(f"  transformers: {transformers.__version__}")

import pandas as pd
print(f"  pandas:       {pd.__version__}")

import fire
print(f"  fire:         {fire.__version__}")

import pyterrier as pt
print(f"  pyterrier:    {pt.__version__}")

from pyterrier_t5 import MonoT5ReRanker
print(f"  pyterrier-t5: OK (MonoT5ReRanker importable)")

import subprocess
result = subprocess.run(['java', '-version'], capture_output=True, text=True)
java_line = (result.stdout + result.stderr).strip().split('\n')[0]
print(f"  java:         {java_line}")
EOF

echo ""
echo "============================================================"
echo "  Environment '$ENV_NAME' is ready."
echo "============================================================"
echo ""
echo "  Activate with:"
echo "    conda activate $ENV_NAME"
echo ""
echo "  Then run the reproduction:"
echo "    bash reproduce_t5_small.sh"
