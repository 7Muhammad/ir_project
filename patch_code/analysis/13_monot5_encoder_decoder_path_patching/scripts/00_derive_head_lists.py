#!/usr/bin/env python3
"""
scripts/00_derive_head_lists.py
=================================
Re-derives configs/heads/encoder_senders.json and decoder_receivers.json
from Experiment 11's and Experiment 3's existing outputs. Does NOT modify
either experiment's outputs — read-only.

Run this only to regenerate/verify the head lists; the checked-in JSON
files are what every other script in this experiment actually reads (see
exp13lib/head_lists.py for the full provenance note on why the 31
decoder receivers are a rank-based top-31, not a 0.02-threshold cut).
"""

from __future__ import annotations

import json
import pathlib

import pandas as pd

EXP_DIR = pathlib.Path(__file__).parent.parent
EXP11_CANONICAL_CSV = (
    EXP_DIR.parent / "11_monot5_encoder_head_patching" / "outputs" / "head_summary_canonical.csv"
)
EXP03_GRID_A_CSV = (
    EXP_DIR.parent / "03_monot5_head_patching_ablation" / "outputs" / "grid_a" / "aggregated" / "head_summary.csv"
)
OUT_DIR = EXP_DIR / "configs" / "heads"

ENCODER_THRESHOLD = 0.02
N_SENDERS = 18
N_RECEIVERS = 31


def derive_senders() -> list[dict]:
    df = pd.read_csv(EXP11_CANONICAL_CSV)
    important = df[df["combined_effect_mean"] > ENCODER_THRESHOLD].sort_values(
        "combined_effect_mean", ascending=False
    )
    if len(important) != N_SENDERS:
        raise ValueError(
            f"Expected {N_SENDERS} encoder senders at threshold {ENCODER_THRESHOLD}, "
            f"got {len(important)} from {EXP11_CANONICAL_CSV}. Report/data may have "
            "changed — investigate before overwriting encoder_senders.json."
        )
    return [
        {
            "layer": int(r.layer),
            "head_idx": int(r.head_idx),
            "label": f"L{int(r.layer)}H{int(r.head_idx)}",
            "combined_effect_mean_canonical": float(r.combined_effect_mean),
        }
        for r in important.itertuples()
    ]


def derive_receivers() -> list[dict]:
    df = pd.read_csv(EXP03_GRID_A_CSV)
    cross = df[df["component"] == "decoder_cross_attn"].sort_values("combined_effect_mean", ascending=False)
    top = cross.head(N_RECEIVERS)
    if len(top) != N_RECEIVERS:
        raise ValueError(f"Expected {N_RECEIVERS} decoder_cross_attn rows, got {len(top)} from {EXP03_GRID_A_CSV}")
    return [
        {
            "layer": int(r.layer),
            "head_idx": int(r.head_idx),
            "component": "decoder_cross_attn",
            "label": f"L{int(r.layer)}-X-H{int(r.head_idx)}",
            "combined_effect_mean_grid_a_n100": float(r.combined_effect_mean),
        }
        for r in top.itertuples()
    ]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    senders = derive_senders()
    receivers = derive_receivers()

    with open(OUT_DIR / "encoder_senders.json", "w", encoding="utf-8") as fh:
        json.dump(senders, fh, indent=2)
    with open(OUT_DIR / "decoder_receivers.json", "w", encoding="utf-8") as fh:
        json.dump(receivers, fh, indent=2)

    print(f"[00_derive_head_lists] wrote {len(senders)} encoder senders -> {OUT_DIR/'encoder_senders.json'}")
    print(f"[00_derive_head_lists] wrote {len(receivers)} decoder receivers -> {OUT_DIR/'decoder_receivers.json'}")
    print(f"[00_derive_head_lists] candidate paths = {len(senders) * len(receivers)}")


if __name__ == "__main__":
    main()
