"""
src/utils.py
============
Shared helpers for the DecoderLens rank experiment: config loading, RNG
seeding, layer naming, and per-attack output-directory layout.

Layer indexing (monoT5-base, 12 encoder blocks)
-----------------------------------------------
    layer 0  = embedding / pre-encoder-block representation
    layer 1  = after encoder block 1
    ...
    layer 12 = after encoder block 12  (matches the normal monoT5 score once
               the encoder final layer norm is applied)

There are therefore ``num_encoder_blocks + 1`` layers (13 for monoT5-base).
"""

from __future__ import annotations

import pathlib
import random
from typing import Dict, List

import yaml


def load_config(config_path: str | pathlib.Path) -> dict:
    """Load the YAML config file into a dict."""
    config_path = pathlib.Path(config_path)
    with open(config_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    return cfg


def seed_everything(seed: int) -> None:
    """Seed Python and torch RNGs for reproducibility."""
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def layer_name(layer_index: int) -> str:
    """
    Human-readable name for an encoder layer index.

        0      -> "embedding"
        1..N   -> "encoder_block_{i}"
    """
    if layer_index == 0:
        return "embedding"
    return f"encoder_block_{layer_index}"


def layer_names(num_layers: int) -> List[str]:
    """Return the list of layer names for indices 0..num_layers-1."""
    return [layer_name(i) for i in range(num_layers)]


# ---------------------------------------------------------------------------
# Per-attack output directory layout
# ---------------------------------------------------------------------------

def attack_output_dirs(base_dir: pathlib.Path, attack_name: str) -> Dict[str, pathlib.Path]:
    """
    Build (and create) the standard output sub-directory layout for one attack:

        outputs/attacks/{attack_name}/
            pairs/
            scores/
            ranks/
            plots/
            sanity/
            status.json   (file, not created here)
    """
    root = base_dir / "attacks" / attack_name
    dirs = {
        "root":   root,
        "pairs":  root / "pairs",
        "scores": root / "scores",
        "ranks":  root / "ranks",
        "plots":  root / "plots",
        "sanity": root / "sanity",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def comparison_output_dir(base_dir: pathlib.Path) -> pathlib.Path:
    """Build (and create) the cross-attack comparison output directory."""
    d = base_dir / "attack_comparison"
    d.mkdir(parents=True, exist_ok=True)
    return d
