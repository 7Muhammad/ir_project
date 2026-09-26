#!/usr/bin/env python3
"""
scripts/04_attack_token_hub_analysis.py
==========================================
Experiment 11, Step 5 -- attack-token-as-hub analysis.

Two falsifiable predictions, stated before running:
  Hub hypothesis:     attack token's outgoing attention to the query span is
                       elevated relative to a matched random-document-token
                       control, CONCENTRATED in the heads Step 1/2 already
                       flagged as causally important (layers 9-11).
  Passive hypothesis: no elevation, or elevation not concentrated in the
                       flagged heads -- contamination happens some other way
                       (e.g. broad residual-stream mixing, not through the
                       attack token's own attention pattern).

Computed for ALL 12 heads at layers 9-11 (not only flagged ones) so
"concentrated in flagged heads" is an actual comparison against an unflagged
control group, not an assumption.

Reuses Experiment 6's attack-token position exposure
(exp6lib.run_utils.ExampleInputs.attack_span_indices) and attention
computation (exp6lib.attention.compute_encoder_attentions) -- not
recomputed independently.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
EXP_DIR = SCRIPT_DIR.parent
EXP1_DIR = EXP_DIR.parent / "01_monot5_layer_patching"
EXP6_DIR = EXP_DIR.parent / "06_monot5_query_doc_patching"
sys.path.insert(0, str(EXP1_DIR))
sys.path.insert(0, str(EXP6_DIR))
sys.path.insert(0, str(EXP_DIR))

import pandas as pd

from src.model_utils import load_monot5, resolve_device

from exp6lib.attention import compute_encoder_attentions
from exp6lib.run_utils import build_example_inputs

from exp11lib.hub_analysis import per_head_outgoing_mass, pick_random_control_positions
from exp11lib.run_utils import load_config, resolve_cfg_path

LAYERS = [9, 10, 11]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Experiment 11 -- Step 5 attack-token-as-hub analysis.")
    p.add_argument("--config", default=str(EXP_DIR / "configs" / "default.yaml"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(pathlib.Path(args.config))
    outputs_base = resolve_cfg_path(cfg, cfg["outputs"]["base_dir"])
    out_dir = outputs_base / "hub_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    seed = cfg["runtime"]["seed"]
    threshold = cfg["flag_threshold"]

    examples_path = outputs_base / "canonical" / "selected_examples.jsonl"
    canonical_csv = outputs_base / "canonical" / "results.csv"
    if not examples_path.exists() or not canonical_csv.exists():
        sys.exit(f"[04_attack_token_hub_analysis] Step 1 outputs not found -- "
                  f"run scripts/01_encoder_head_patch_canonical.py first.")
    examples = [json.loads(l) for l in open(examples_path, encoding="utf-8")]
    canonical_df = pd.read_csv(canonical_csv)

    flagged_slots = set()
    causal_by_slot = canonical_df[canonical_df["layer"].isin(LAYERS)].groupby(
        ["layer", "head_idx"]
    )["combined_effect"].mean()
    for (layer, head), effect in causal_by_slot.items():
        if effect > threshold:
            flagged_slots.add((int(layer), int(head)))
    print(f"[04_attack_token_hub_analysis] {len(flagged_slots)} flagged (layer,head) slots "
          f"at layers {LAYERS}: {sorted(flagged_slots)}")

    device = resolve_device(cfg["model"]["device"])
    model, tokenizer = load_monot5(cfg["model"]["checkpoint"], device)
    max_length = cfg["model"]["max_length"]

    all_rows = []
    n_used = 0
    for i, ex in enumerate(examples):
        inputs = build_example_inputs(tokenizer, ex, max_length, device)
        if inputs.status != "ok" or not inputs.attack_span_indices:
            continue

        atk_attn = compute_encoder_attentions(model, inputs.attack_enc, device)
        example_key = f"{ex['qid']}:{ex['docid']}"
        control_positions = pick_random_control_positions(
            inputs.doc_span_control_attack, inputs.attack_span_indices,
            n=len(inputs.attack_span_indices), seed=seed, example_key=example_key,
        )
        if not control_positions:
            continue

        for layer in LAYERS:
            attack_rows = per_head_outgoing_mass(
                atk_attn[layer], inputs.attack_span_indices, inputs.query_span, inputs.doc_span_control_attack,
            )
            control_rows = per_head_outgoing_mass(
                atk_attn[layer], control_positions, inputs.query_span, inputs.doc_span_control_attack,
            )
            for a_row, c_row in zip(attack_rows, control_rows):
                all_rows.append({
                    "qid": ex["qid"], "docid": ex["docid"], "layer": layer, "head": a_row["head"],
                    "attack_to_query": a_row["outgoing_to_query"], "attack_to_doc": a_row["outgoing_to_doc"],
                    "control_to_query": c_row["outgoing_to_query"], "control_to_doc": c_row["outgoing_to_doc"],
                })
        n_used += 1
        if (i + 1) % 20 == 0 or i == len(examples) - 1:
            print(f"    example {i+1}/{len(examples)} (used={n_used})", flush=True)

    df = pd.DataFrame(all_rows)
    df.to_csv(out_dir / "results.csv", index=False)
    print(f"[04_attack_token_hub_analysis] wrote {len(df)} rows ({n_used} examples) -> {out_dir/'results.csv'}")

    df["elevation_to_query"] = df["attack_to_query"] - df["control_to_query"]
    agg = df.groupby(["layer", "head"]).agg(
        n=("qid", "size"),
        attack_to_query_mean=("attack_to_query", "mean"),
        control_to_query_mean=("control_to_query", "mean"),
        elevation_to_query_mean=("elevation_to_query", "mean"),
    ).reset_index()
    agg["flagged"] = agg.apply(lambda r: (int(r["layer"]), int(r["head"])) in flagged_slots, axis=1)
    agg.to_csv(out_dir / "aggregated.csv", index=False)

    flagged_elev = agg.loc[agg["flagged"], "elevation_to_query_mean"]
    unflagged_elev = agg.loc[~agg["flagged"], "elevation_to_query_mean"]
    flagged_mean = float(flagged_elev.mean()) if len(flagged_elev) else float("nan")
    unflagged_mean = float(unflagged_elev.mean()) if len(unflagged_elev) else float("nan")

    print("\n[04_attack_token_hub_analysis] Elevation (attack_token -> query, vs. matched random doc token):")
    print(agg.sort_values("elevation_to_query_mean", ascending=False).round(4).to_string(index=False))

    hub_holds = (flagged_mean > 0) and (len(flagged_elev) == 0 or flagged_mean > unflagged_mean)
    verdict = (
        f"HUB HYPOTHESIS {'HOLDS' if hub_holds else 'DOES NOT HOLD'}: "
        f"flagged heads' mean elevation = {flagged_mean:+.4f} vs. unflagged heads' "
        f"{unflagged_mean:+.4f} ({len(flagged_elev)} flagged / {len(unflagged_elev)} unflagged "
        f"(layer,head) cells at layers {LAYERS})."
    )
    print(f"\n{verdict}")
    (out_dir / "verdict.txt").write_text(verdict + "\n", encoding="utf-8")
    print(f"\n  Next: python scripts/05_aggregate_and_plot.py --config {args.config}")


if __name__ == "__main__":
    main()
