#!/usr/bin/env python3
"""
scripts/06_run_iid_test.py
=============================
Held-out (pair-disjoint) IID attack evaluation on the TEST split (task spec
section 9), plus clean-input damage evaluation on TEST clean pairs (section
10) — for every candidate, at its validation-selected scale AND the fixed
reference scale (1.5, deduplicated if equal), never re-selecting or tuning
anything on test data itself.

Outputs:
  outputs/test/iid_test_per_example.csv
  outputs/test/iid_summary.csv          (per-candidate x scale_role: mean/median/std recovery, frac>0, frac>=1, global_recovery)
  outputs/test/clean_damage_per_example.csv
  outputs/test/clean_damage_summary.csv (per-candidate x scale_role)
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

from exp14lib.engine import compute_clean_damage_rows, compute_defense_rows_selected_scales  # noqa: E402
from exp14lib.head_lists import (  # noqa: E402
    apply_head_caps,
    decoder_heads_from_manifest,
    encoder_heads_from_manifest,
    load_candidate_heads,
)
from exp14lib.metrics import clean_damage_summary, global_recovery, recovery_summary  # noqa: E402
from exp14lib.run_utils import is_already_successful, load_config, resolve_cfg_path, write_status  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Held-out IID test + clean-damage evaluation.")
    p.add_argument("--config", default=str(EXP_DIR / "configs" / "default.yaml"))
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def make_scale_lookup(selected_scales: dict, reference_scale: float):
    def get_scales_for(side, layer, head_idx, condition):
        key = f"{side}:{int(layer)}:{int(head_idx)}:{condition}"
        entry = selected_scales.get(key)
        if entry is None:
            return []
        scales = {entry["selected_scale"], reference_scale}
        return sorted(scales)
    return get_scales_for


def tag_scale_role(df: pd.DataFrame, selected_scales: dict, reference_scale: float) -> pd.DataFrame:
    def role(row):
        key = f"{row['side']}:{int(row['layer'])}:{int(row['head_idx'])}:{row['condition']}"
        selected = selected_scales.get(key, {}).get("selected_scale")
        roles = []
        if row["scale"] == selected:
            roles.append("selected")
        if row["scale"] == reference_scale:
            roles.append("reference")
        return "+".join(roles) if roles else "other"
    df = df.copy()
    df["scale_role"] = df.apply(role, axis=1)
    return df


def summarize(df: pd.DataFrame, value_fn) -> pd.DataFrame:
    rows = []
    for (side, layer, head_idx, condition, scale_role), group in df.groupby(
        ["side", "layer", "head_idx", "condition", "scale_role"]
    ):
        rows.append({
            "side": side, "layer": layer, "head_idx": head_idx, "condition": condition,
            "scale_role": scale_role, **value_fn(group),
        })
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    cfg = load_config(pathlib.Path(args.config))
    outputs_base = resolve_cfg_path(cfg, cfg["outputs"]["base_dir"])
    test_dir = outputs_base / "test"

    required = ["iid_test_per_example.csv", "iid_summary.csv", "clean_damage_per_example.csv", "clean_damage_summary.csv"]
    if not args.force and is_already_successful(test_dir, required):
        print("[06_run_iid_test] RESUME: already done — skipping.")
        return

    manifest = load_candidate_heads(outputs_base / "candidate_heads.json")
    manifest = apply_head_caps(manifest, cfg["heads"].get("max_encoder_heads"), cfg["heads"].get("max_decoder_heads"))
    encoder_heads = encoder_heads_from_manifest(manifest)
    decoder_heads = decoder_heads_from_manifest(manifest)

    reference_scale = cfg["intervention"]["reference_scale"]
    with open(outputs_base / "scale_selection" / "selected_scales.json", encoding="utf-8") as fh:
        selected_scales = json.load(fh)
    get_scales_for = make_scale_lookup(selected_scales, reference_scale)

    directions = load_directions(outputs_base / "directions" / "iid" / "directions.pt")

    with open(outputs_base / "baselines" / "test_by_attack.json", encoding="utf-8") as fh:
        by_attack = json.load(fh)
    flat_examples = [(attack, ex) for attack, exs in by_attack.items() for ex in exs]

    with open(outputs_base / "baselines" / "test_clean_examples.json", encoding="utf-8") as fh:
        clean_examples = json.load(fh)

    device = resolve_device(cfg["model"]["device"])
    model, tokenizer = load_monot5(cfg["model"]["checkpoint"], device)
    true_id, false_id = get_true_false_token_ids(tokenizer)

    status = {"status": "failed"}
    try:
        rows, skip = compute_defense_rows_selected_scales(
            model, tokenizer, flat_examples, encoder_heads, decoder_heads, directions, get_scales_for,
            cfg["model"]["max_length"], true_id, false_id, device,
        )
        test_df = pd.DataFrame(rows)
        test_df = tag_scale_role(test_df, selected_scales, reference_scale)
        test_dir.mkdir(parents=True, exist_ok=True)
        test_df.to_csv(test_dir / "iid_test_per_example.csv", index=False)

        def recov_stats(group):
            stats = recovery_summary(list(group["recovery"]))
            stats["global_recovery"] = global_recovery(
                list(zip(group["score_attack"], group["score_control"], group["score_defended"]))
            )
            return stats

        iid_summary = summarize(test_df, recov_stats)
        iid_summary.to_csv(test_dir / "iid_summary.csv", index=False)

        clean_rows, clean_skip = compute_clean_damage_rows(
            model, tokenizer, clean_examples, encoder_heads, decoder_heads, directions, get_scales_for,
            cfg["model"]["max_length"], true_id, false_id, device,
        )
        clean_df = pd.DataFrame(clean_rows)
        clean_df = tag_scale_role(clean_df, selected_scales, reference_scale)
        clean_df.to_csv(test_dir / "clean_damage_per_example.csv", index=False)

        thresholds = cfg["clean_damage"]["thresholds"]

        def damage_stats(group):
            return clean_damage_summary(list(zip(group["score_clean"], group["score_clean_defended"])), thresholds)

        clean_summary = summarize(clean_df, damage_stats)
        clean_summary.to_csv(test_dir / "clean_damage_summary.csv", index=False)

        status.update({
            "n_test_examples": len(flat_examples), "n_iid_rows": len(rows), "iid_skip": skip,
            "n_clean_examples": len(clean_examples), "n_clean_rows": len(clean_rows), "clean_skip": clean_skip,
            "status": "success",
        })
        print(f"[06_run_iid_test] IID: {len(flat_examples)} test examples -> {len(rows)} rows "
              f"({len(iid_summary)} summary rows). skip={skip}")
        print(f"[06_run_iid_test] Clean damage: {len(clean_examples)} test clean pairs -> {len(clean_rows)} rows "
              f"({len(clean_summary)} summary rows). skip={clean_skip}")
    finally:
        write_status(test_dir, status)

    if status["status"] != "success":
        sys.exit(1)


if __name__ == "__main__":
    main()
