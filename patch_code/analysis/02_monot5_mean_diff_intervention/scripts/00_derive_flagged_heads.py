#!/usr/bin/env python3
"""
scripts/00_derive_flagged_heads.py
====================================
Derive Part 2's intervention scope from Experiment 3's grid_a aggregate and
write it to outputs/flagged_heads.json.

No pre-existing "flagged heads" artifact exists in Experiment 3 — this reads
its per-head summary CSV and filters by combined_effect_mean, per user
decision (2026-07-12): grid_a, threshold 0.02 -> 22 heads.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
EXP_DIR = SCRIPT_DIR.parent
EXP1_DIR = EXP_DIR.parent / "01_monot5_layer_patching"
EXP3_DIR = EXP_DIR.parent / "03_monot5_head_patching_ablation"
sys.path.insert(0, str(EXP1_DIR))
sys.path.insert(0, str(EXP3_DIR))
sys.path.insert(0, str(EXP_DIR))

from headlib.run_utils import load_config, resolve_cfg_path  # noqa: E402
from exp2lib.flagged_heads import load_flagged_heads  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Derive Experiment 2's flagged-head scope.")
    p.add_argument("--config", default=str(EXP_DIR / "configs" / "default.yaml"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(pathlib.Path(args.config))
    fh_cfg = cfg["flagged_heads"]

    source_csv = resolve_cfg_path(cfg, fh_cfg["source_csv"])
    flagged = load_flagged_heads(source_csv, fh_cfg["combined_effect_threshold"])
    max_heads = fh_cfg.get("max_heads")
    if max_heads:
        flagged = flagged[:max_heads]

    outputs_base = resolve_cfg_path(cfg, cfg["outputs"]["base_dir"])
    outputs_base.mkdir(parents=True, exist_ok=True)
    out_path = outputs_base / "flagged_heads.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(flagged, fh, indent=2)

    n_cross = sum(1 for h in flagged if h["component"] == "decoder_cross_attn")
    n_self = sum(1 for h in flagged if h["component"] == "decoder_self_attn")
    print(f"[00_derive_flagged_heads] {len(flagged)} heads "
          f"({n_cross} decoder_cross_attn, {n_self} decoder_self_attn) "
          f"from {source_csv} (threshold={fh_cfg['combined_effect_threshold']})")
    print(f"[00_derive_flagged_heads] Wrote {out_path}")


if __name__ == "__main__":
    main()
