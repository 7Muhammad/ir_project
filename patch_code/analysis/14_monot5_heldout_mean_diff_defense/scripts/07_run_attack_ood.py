#!/usr/bin/env python3
"""
scripts/07_run_attack_ood.py
===============================
Attack-OOD generalization (task spec section 11): leave-one-token-out (7
folds), leave-one-position-out (3 folds), leave-one-repetition-out (5
folds). For each fold: fit a direction using ONLY seen-attack TRAIN
examples, select a scale using ONLY seen-attack VALIDATION examples, then
evaluate on held-out-attack TEST examples (which are additionally
pair-held-out via the existing split — the fold only removes attack
CONFIGURATIONS, never touches which pairs are visible). No held-out attack
configuration ever enters direction fitting or scale selection for its own
fold (task spec section 21).

Per-fold outputs are resume-safe (own status.json), then concatenated:
  outputs/ood_folds_manifest.json
  outputs/directions/leave_{token,position,repetition}_out/{value}/directions.pt
  outputs/ood_test_per_example.csv       (all folds, all factors, tagged)
  outputs/ood_summary.csv                (per candidate x factor, pooled across that factor's folds)
  outputs/ood_clean_damage_per_example.csv
  outputs/ood_clean_damage_summary.csv
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

from exp2lib.run_utils import save_directions  # noqa: E402

from exp14lib.attacks import list_all_attacks  # noqa: E402
from exp14lib.data_pool import pool_round_robin  # noqa: E402
from exp14lib.direction_fit import direction_norm_rows, fit_directions_train_only  # noqa: E402
from exp14lib.engine import compute_clean_damage_rows, compute_defense_rows_for_examples, compute_defense_rows_selected_scales  # noqa: E402
from exp14lib.head_lists import (  # noqa: E402
    apply_head_caps,
    decoder_heads_from_manifest,
    encoder_heads_from_manifest,
    load_candidate_heads,
)
from exp14lib.metrics import global_recovery, recovery_summary  # noqa: E402
from exp14lib.ood_folds import build_all_folds, folds_to_manifest  # noqa: E402
from exp14lib.run_utils import is_already_successful, load_config, resolve_cfg_path, write_status  # noqa: E402
from exp14lib.scale_selection import select_scales  # noqa: E402

FACTOR_DIR_NAME = {"token": "leave_token_out", "position": "leave_position_out", "repetitions": "leave_repetition_out"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Attack-OOD evaluation (leave-one-{token,position,repetition}-out).")
    p.add_argument("--config", default=str(EXP_DIR / "configs" / "default.yaml"))
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def make_scale_lookup(selected: dict, reference_scale: float):
    def get_scales_for(side, layer, head_idx, condition):
        key = f"{side}:{int(layer)}:{int(head_idx)}:{condition}"
        entry = selected.get(key)
        if entry is None:
            return []
        return sorted({entry["selected_scale"], reference_scale})
    return get_scales_for


def run_one_fold(
    factor, value, seen_attacks, held_out_attacks,
    model, tokenizer, true_id, false_id, device, cfg,
    encoder_heads, decoder_heads,
    train_by_attack, validation_by_attack, test_by_attack, test_clean_examples,
    outputs_base, force,
):
    factor_dir = FACTOR_DIR_NAME[factor]
    fold_dir = outputs_base / "ood" / factor_dir / value
    directions_fold_dir = outputs_base / "directions" / factor_dir / value

    required = ["ood_test_per_example.csv", "clean_damage_per_example.csv", "scale_selection.json"]
    if not force and is_already_successful(fold_dir, required):
        print(f"    [RESUME] {factor}={value} — skipping.")
        return

    seen_set = set(seen_attacks)
    held_out_set = set(held_out_attacks)
    max_length = cfg["model"]["max_length"]
    scales = list(cfg["intervention"]["scales"])
    reference_scale = cfg["intervention"]["reference_scale"]
    if reference_scale not in scales:
        scales.append(reference_scale)

    status = {"status": "failed", "factor": factor, "held_out_value": value}
    try:
        # 1. Fit direction on seen-attack TRAIN examples only.
        train_seen = {a: exs for a, exs in train_by_attack.items() if a in seen_set}
        pooled_train = pool_round_robin(train_seen, cap=cfg["sampling"]["max_train_examples_total"])
        directions, counts, fit_stats = fit_directions_train_only(
            model, tokenizer, pooled_train, encoder_heads, decoder_heads, max_length, device, true_id, false_id,
        )
        directions_fold_dir.mkdir(parents=True, exist_ok=True)
        save_directions(directions, directions_fold_dir / "directions.pt")
        norm_rows = direction_norm_rows(directions, counts, tag=f"{factor}={value}")
        pd.DataFrame(norm_rows).to_csv(directions_fold_dir / "direction_norms.csv", index=False)

        # 2. Select scale on seen-attack VALIDATION examples only.
        val_seen_flat = [(a, ex) for a, exs in validation_by_attack.items() if a in seen_set for ex in exs]
        val_rows, val_skip = compute_defense_rows_for_examples(
            model, tokenizer, val_seen_flat, encoder_heads, decoder_heads, directions, scales,
            max_length, true_id, false_id, device,
        )
        val_df = pd.DataFrame(val_rows)
        selected = select_scales(val_df) if len(val_df) else {}
        fold_dir.mkdir(parents=True, exist_ok=True)
        with open(fold_dir / "scale_selection.json", "w", encoding="utf-8") as fh:
            json.dump(selected, fh, indent=2)

        # 3. Evaluate on held-out-attack TEST examples (already pair-held-out).
        get_scales_for = make_scale_lookup(selected, reference_scale)
        test_held_out_flat = [(a, ex) for a, exs in test_by_attack.items() if a in held_out_set for ex in exs]
        test_rows, test_skip = compute_defense_rows_selected_scales(
            model, tokenizer, test_held_out_flat, encoder_heads, decoder_heads, directions, get_scales_for,
            max_length, true_id, false_id, device,
        )
        test_df = pd.DataFrame(test_rows)
        if len(test_df):
            test_df["ood_factor"] = factor
            test_df["held_out_value"] = value
        test_df.to_csv(fold_dir / "ood_test_per_example.csv", index=False)

        # 4. Clean-score damage for this fold's fixed direction.
        clean_rows, clean_skip = compute_clean_damage_rows(
            model, tokenizer, test_clean_examples, encoder_heads, decoder_heads, directions, get_scales_for,
            max_length, true_id, false_id, device,
        )
        clean_df = pd.DataFrame(clean_rows)
        if len(clean_df):
            clean_df["ood_factor"] = factor
            clean_df["held_out_value"] = value
        clean_df.to_csv(fold_dir / "clean_damage_per_example.csv", index=False)

        status.update({
            "n_seen_attacks": len(seen_attacks), "n_held_out_attacks": len(held_out_attacks),
            "fit_stats": fit_stats, "n_val_rows": len(val_rows), "val_skip": val_skip,
            "n_test_rows": len(test_rows), "test_skip": test_skip,
            "n_clean_rows": len(clean_rows), "clean_skip": clean_skip,
            "status": "success",
        })
        print(f"    [DONE] {factor}={value}: {len(seen_attacks)} seen / {len(held_out_attacks)} held-out attacks, "
              f"{len(test_rows)} test rows, {len(clean_rows)} clean rows.")
    finally:
        write_status(fold_dir, status)


def main() -> None:
    args = parse_args()
    cfg = load_config(pathlib.Path(args.config))
    outputs_base = resolve_cfg_path(cfg, cfg["outputs"]["base_dir"])

    manifest = load_candidate_heads(outputs_base / "candidate_heads.json")
    manifest = apply_head_caps(manifest, cfg["heads"].get("max_encoder_heads"), cfg["heads"].get("max_decoder_heads"))
    encoder_heads = encoder_heads_from_manifest(manifest)
    decoder_heads = decoder_heads_from_manifest(manifest)

    all_specs = list_all_attacks(cfg)
    all_folds = build_all_folds(all_specs)
    with open(outputs_base / "ood_folds_manifest.json", "w", encoding="utf-8") as fh:
        json.dump(folds_to_manifest(all_folds), fh, indent=2)

    with open(outputs_base / "baselines" / "train_by_attack.json", encoding="utf-8") as fh:
        train_by_attack = json.load(fh)
    with open(outputs_base / "baselines" / "validation_by_attack.json", encoding="utf-8") as fh:
        validation_by_attack = json.load(fh)
    with open(outputs_base / "baselines" / "test_by_attack.json", encoding="utf-8") as fh:
        test_by_attack = json.load(fh)
    with open(outputs_base / "baselines" / "test_clean_examples.json", encoding="utf-8") as fh:
        test_clean_examples = json.load(fh)

    device = resolve_device(cfg["model"]["device"])
    model, tokenizer = load_monot5(cfg["model"]["checkpoint"], device)
    true_id, false_id = get_true_false_token_ids(tokenizer)

    factors = cfg["ood"]["factors"]
    for factor in factors:
        print(f"[07_run_attack_ood] factor={factor}: {len(all_folds[factor])} folds")
        for fold in all_folds[factor]:
            run_one_fold(
                factor, fold.held_out_value, fold.seen_attacks, fold.held_out_attacks,
                model, tokenizer, true_id, false_id, device, cfg,
                encoder_heads, decoder_heads,
                train_by_attack, validation_by_attack, test_by_attack, test_clean_examples,
                outputs_base, args.force,
            )

    # --- Concatenate + aggregate across folds, per factor ---
    test_frames, clean_frames = [], []
    for factor in factors:
        factor_dir_name = FACTOR_DIR_NAME[factor]
        for fold in all_folds[factor]:
            fold_dir = outputs_base / "ood" / factor_dir_name / fold.held_out_value
            t = pd.read_csv(fold_dir / "ood_test_per_example.csv") if (fold_dir / "ood_test_per_example.csv").exists() else pd.DataFrame()
            c = pd.read_csv(fold_dir / "clean_damage_per_example.csv") if (fold_dir / "clean_damage_per_example.csv").exists() else pd.DataFrame()
            if len(t):
                test_frames.append(t)
            if len(c):
                clean_frames.append(c)

    all_test = pd.concat(test_frames, ignore_index=True) if test_frames else pd.DataFrame()
    all_clean = pd.concat(clean_frames, ignore_index=True) if clean_frames else pd.DataFrame()
    all_test.to_csv(outputs_base / "ood_test_per_example.csv", index=False)
    all_clean.to_csv(outputs_base / "ood_clean_damage_per_example.csv", index=False)

    summary_rows = []
    if len(all_test):
        for (side, layer, head_idx, condition, ood_factor), group in all_test.groupby(
            ["side", "layer", "head_idx", "condition", "ood_factor"]
        ):
            stats = recovery_summary(list(group["recovery"]))
            stats["global_recovery"] = global_recovery(
                list(zip(group["score_attack"], group["score_control"], group["score_defended"]))
            )
            summary_rows.append({
                "side": side, "layer": layer, "head_idx": head_idx, "condition": condition,
                "ood_factor": ood_factor, **stats,
            })
    pd.DataFrame(summary_rows).to_csv(outputs_base / "ood_summary.csv", index=False)

    clean_summary_rows = []
    if len(all_clean):
        from exp14lib.metrics import clean_damage_summary
        thresholds = cfg["clean_damage"]["thresholds"]
        for (side, layer, head_idx, condition, ood_factor), group in all_clean.groupby(
            ["side", "layer", "head_idx", "condition", "ood_factor"]
        ):
            stats = clean_damage_summary(list(zip(group["score_clean"], group["score_clean_defended"])), thresholds)
            clean_summary_rows.append({
                "side": side, "layer": layer, "head_idx": head_idx, "condition": condition,
                "ood_factor": ood_factor, **stats,
            })
    pd.DataFrame(clean_summary_rows).to_csv(outputs_base / "ood_clean_damage_summary.csv", index=False)

    print(f"[07_run_attack_ood] {len(all_test)} OOD test rows -> ood_test_per_example.csv "
          f"({len(summary_rows)} summary rows)")
    print(f"[07_run_attack_ood] {len(all_clean)} OOD clean rows -> ood_clean_damage_per_example.csv "
          f"({len(clean_summary_rows)} summary rows)")


if __name__ == "__main__":
    main()
