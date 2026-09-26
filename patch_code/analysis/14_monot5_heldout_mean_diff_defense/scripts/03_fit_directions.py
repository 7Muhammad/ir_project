#!/usr/bin/env python3
"""
scripts/03_fit_directions.py
===============================
Fit ONE fixed 64-dim mean-diff direction per candidate head (18 encoder
self-attn + 31 decoder cross-attn), using ONLY the round-robin-pooled TRAIN
examples cached by 02_cache_baselines.py. Directions are frozen here —
validation/test never contribute (task spec section 5, DECISIONS.md item 2).

Requires a GPU-capable machine for anything beyond the smoke config (each
example costs 2 forward passes: control + attack). Not SLURM-heavy by
itself (one pass per example, no per-head/per-scale multiplication yet —
that happens at intervention time in scripts 04/06/07).

Outputs:
  outputs/directions/iid/directions.pt          (DirKey -> Tensor, CPU)
  outputs/directions/iid/direction_norms.csv
  outputs/directions/iid/status.json
"""

from __future__ import annotations

import argparse
import pathlib
import sys

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
EXP_DIR = SCRIPT_DIR.parent
EXP1_DIR = EXP_DIR.parent / "01_monot5_layer_patching"
EXP2_DIR = EXP_DIR.parent / "02_monot5_mean_diff_intervention"
EXP3_DIR = EXP_DIR.parent / "03_monot5_head_patching_ablation"
sys.path.insert(0, str(EXP1_DIR))
sys.path.insert(0, str(EXP2_DIR))
sys.path.insert(0, str(EXP3_DIR))
sys.path.insert(0, str(EXP_DIR))

import json  # noqa: E402

from src.model_utils import get_true_false_token_ids, load_monot5, resolve_device  # noqa: E402

from exp2lib.run_utils import save_directions, save_norm_rows  # noqa: E402

from exp14lib.direction_fit import direction_norm_rows, fit_directions_train_only  # noqa: E402
from exp14lib.head_lists import (  # noqa: E402
    apply_head_caps,
    decoder_heads_from_manifest,
    encoder_heads_from_manifest,
    load_candidate_heads,
)
from exp14lib.run_utils import is_already_successful, load_config, resolve_cfg_path, write_status  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fit train-only mean-diff directions for candidate heads.")
    p.add_argument("--config", default=str(EXP_DIR / "configs" / "default.yaml"))
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(pathlib.Path(args.config))
    outputs_base = resolve_cfg_path(cfg, cfg["outputs"]["base_dir"])
    directions_dir = outputs_base / "directions" / "iid"

    required = ["directions.pt", "direction_norms.csv"]
    if not args.force and is_already_successful(directions_dir, required):
        print("[03_fit_directions] RESUME: already done — skipping.")
        return

    manifest = load_candidate_heads(outputs_base / "candidate_heads.json")
    manifest = apply_head_caps(
        manifest,
        cfg["heads"].get("max_encoder_heads"),
        cfg["heads"].get("max_decoder_heads"),
    )
    encoder_heads = encoder_heads_from_manifest(manifest)
    decoder_heads = decoder_heads_from_manifest(manifest)

    with open(outputs_base / "baselines" / "train_pooled_examples.json", encoding="utf-8") as fh:
        train_examples = json.load(fh)

    device = resolve_device(cfg["model"]["device"])
    model, tokenizer = load_monot5(cfg["model"]["checkpoint"], device)
    true_id, false_id = get_true_false_token_ids(tokenizer)

    status = {"status": "failed"}
    try:
        directions, counts, stats = fit_directions_train_only(
            model, tokenizer, train_examples, encoder_heads, decoder_heads,
            cfg["model"]["max_length"], device, true_id, false_id,
        )
        directions_dir.mkdir(parents=True, exist_ok=True)
        save_directions(directions, directions_dir / "directions.pt")
        norm_rows = direction_norm_rows(directions, counts, tag="iid")
        save_norm_rows(norm_rows, directions_dir / "direction_norms.csv")

        missing = stats["n_directions_expected"] - stats["n_directions_found"]
        status.update(stats)
        status["missing_heads"] = missing
        status["status"] = "success"
        print(f"[03_fit_directions] fit {stats['n_directions_found']}/{stats['n_directions_expected']} "
              f"candidate-head directions from {stats['n_used']} train examples "
              f"({stats['n_align_failed']} align-failed, {stats['n_skipped_epsilon']} skipped-epsilon).")
        if missing:
            print(f"[03_fit_directions] WARNING: {missing} candidate heads have NO direction "
                  "(no common control+attack example reached them) — downstream scripts must skip them explicitly.")
    finally:
        write_status(directions_dir, status)

    if status["status"] != "success":
        sys.exit(1)


if __name__ == "__main__":
    main()
