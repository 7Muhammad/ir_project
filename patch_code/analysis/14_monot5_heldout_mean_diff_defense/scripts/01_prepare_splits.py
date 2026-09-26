#!/usr/bin/env python3
"""
scripts/01_prepare_splits.py
===============================
Build the deterministic, pair-disjoint train/validation/test split from
Experiment 1's canonical pair pool (outputs/pairs/pairs.jsonl), BEFORE any
attack-success filtering (see exp14lib/splits.py and DECISIONS.md item 2).

Output: outputs/split_manifest.json
"""

from __future__ import annotations

import argparse
import pathlib
import sys

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
EXP_DIR = SCRIPT_DIR.parent
EXP1_DIR = EXP_DIR.parent / "01_monot5_layer_patching"
EXP3_DIR = EXP_DIR.parent / "03_monot5_head_patching_ablation"
sys.path.insert(0, str(EXP1_DIR))
sys.path.insert(0, str(EXP3_DIR))
sys.path.insert(0, str(EXP_DIR))

from src.data_utils import load_pairs  # noqa: E402

from exp14lib.run_utils import load_config, resolve_cfg_path  # noqa: E402
from exp14lib.splits import build_split, save_split_manifest  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build the pair-disjoint train/validation/test split.")
    p.add_argument("--config", default=str(EXP_DIR / "configs" / "default.yaml"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(pathlib.Path(args.config))

    pairs_path = resolve_cfg_path(cfg, cfg["data"]["pairs_jsonl"])
    pairs = load_pairs(pairs_path)

    seed = cfg["split"]["seed"]
    ratios = cfg["split"]["ratios"]
    split = build_split(pairs, seed=seed, ratios=ratios)

    outputs_base = resolve_cfg_path(cfg, cfg["outputs"]["base_dir"])
    out_path = outputs_base / "split_manifest.json"
    save_split_manifest(split, seed, ratios, out_path)

    print(f"[01_prepare_splits] {len(pairs)} unique pairs from {pairs_path}")
    for name in ("train", "validation", "test"):
        print(f"[01_prepare_splits]   {name:10s}: {len(split[name])} pairs")
    print(f"[01_prepare_splits] wrote {out_path}")


if __name__ == "__main__":
    main()
