#!/usr/bin/env python3
"""
scripts/04_run_validation.py
===============================
Sweep every configured scale, for every candidate (decoder head; or
encoder head x position-mask condition), on the VALIDATION split only
(task spec section 8). Never touches TEST data. Frozen TRAIN-only
directions from 03_fit_directions.py.

Output: outputs/validation_results.csv (one row per example x candidate x
scale), outputs/validation/status.json.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
EXP_DIR = SCRIPT_DIR.parent
EXP1_DIR = EXP_DIR.parent / "01_monot5_layer_patching"
EXP2_DIR = EXP_DIR.parent / "02_monot5_mean_diff_intervention"
EXP3_DIR = EXP_DIR.parent / "03_monot5_head_patching_ablation"
EXP6_DIR = EXP_DIR.parent / "06_monot5_query_doc_patching"
sys.path.insert(0, str(EXP1_DIR))
sys.path.insert(0, str(EXP2_DIR))
sys.path.insert(0, str(EXP3_DIR))
sys.path.insert(0, str(EXP6_DIR))
sys.path.insert(0, str(EXP_DIR))

import pandas as pd  # noqa: E402

from src.model_utils import get_true_false_token_ids, load_monot5, resolve_device  # noqa: E402

from exp2lib.run_utils import load_directions  # noqa: E402

from exp14lib.engine import compute_clean_damage_rows_multiscale, compute_defense_rows_for_examples  # noqa: E402
from exp14lib.head_lists import (  # noqa: E402
    apply_head_caps,
    decoder_heads_from_manifest,
    encoder_heads_from_manifest,
    load_candidate_heads,
)
from exp14lib.run_utils import is_already_successful, load_config, resolve_cfg_path, write_status  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sweep scales on the validation split for every candidate.")
    p.add_argument("--config", default=str(EXP_DIR / "configs" / "default.yaml"))
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(pathlib.Path(args.config))
    outputs_base = resolve_cfg_path(cfg, cfg["outputs"]["base_dir"])
    validation_dir = outputs_base / "validation"

    if not args.force and is_already_successful(
        validation_dir, ["validation_results.csv", "validation_clean_damage.csv"]
    ):
        print("[04_run_validation] RESUME: already done — skipping.")
        return

    manifest = load_candidate_heads(outputs_base / "candidate_heads.json")
    manifest = apply_head_caps(manifest, cfg["heads"].get("max_encoder_heads"), cfg["heads"].get("max_decoder_heads"))
    encoder_heads = encoder_heads_from_manifest(manifest)
    decoder_heads = decoder_heads_from_manifest(manifest)

    scales = list(cfg["intervention"]["scales"])
    ref_scale = cfg["intervention"]["reference_scale"]
    if ref_scale not in scales:
        scales.append(ref_scale)

    directions = load_directions(outputs_base / "directions" / "iid" / "directions.pt")

    with open(outputs_base / "baselines" / "validation_by_attack.json", encoding="utf-8") as fh:
        by_attack = json.load(fh)
    flat_examples = [(attack, ex) for attack, exs in by_attack.items() for ex in exs]

    with open(outputs_base / "baselines" / "validation_clean_examples.json", encoding="utf-8") as fh:
        clean_examples = json.load(fh)

    device = resolve_device(cfg["model"]["device"])
    model, tokenizer = load_monot5(cfg["model"]["checkpoint"], device)
    true_id, false_id = get_true_false_token_ids(tokenizer)

    status = {"status": "failed"}
    try:
        rows, skip = compute_defense_rows_for_examples(
            model, tokenizer, flat_examples, encoder_heads, decoder_heads, directions, scales,
            cfg["model"]["max_length"], true_id, false_id, device,
        )
        df = pd.DataFrame(rows)
        validation_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(validation_dir / "validation_results.csv", index=False)

        clean_rows, clean_skip = compute_clean_damage_rows_multiscale(
            model, tokenizer, clean_examples, encoder_heads, decoder_heads, directions, scales,
            cfg["model"]["max_length"], true_id, false_id, device,
        )
        clean_df = pd.DataFrame(clean_rows)
        clean_df.to_csv(validation_dir / "validation_clean_damage.csv", index=False)

        status.update({
            "n_examples": len(flat_examples), "n_rows": len(rows), "scales": scales,
            "skip_counts": skip,
            "n_clean_examples": len(clean_examples), "n_clean_rows": len(clean_rows), "clean_skip_counts": clean_skip,
            "status": "success",
        })
        print(f"[04_run_validation] {len(flat_examples)} validation examples x "
              f"{manifest['n_total_candidates']} candidates x {len(scales)} scales -> {len(rows)} rows "
              f"(skipped: {skip})")
        print(f"[04_run_validation] {len(clean_examples)} validation clean pairs x "
              f"{manifest['n_total_candidates']} candidates x {len(scales)} scales -> {len(clean_rows)} clean rows "
              f"(skipped: {clean_skip})")
    finally:
        write_status(validation_dir, status)

    if status["status"] != "success":
        sys.exit(1)


if __name__ == "__main__":
    main()
