#!/usr/bin/env python3
"""
scripts/03_run_layer_patching.py
=================================
Stage 03: Run layer-level activation patching for all selected examples.

Input
-----
  outputs/scores/selected_examples.jsonl   (from Stage 01)
  outputs/activations/{qid}_{docid}_control.pt  (from Stage 02)
  outputs/activations/{qid}_{docid}_attack.pt   (from Stage 02)

Outputs
-------
  outputs/patching/layer_patching_results_detailed.csv  — per-example results
  outputs/patching/layer_patching_results.csv           — aggregated results

What this script does
----------------------
For each selected (query, passage) example, for each (layer, component) pair,
we perform two patching experiments:

  FORWARD patch:
    Run the CONTROL input but replace one layer-component's hidden state
    with the corresponding hidden state from the ATTACK run.
    → Measures: how much of the attack signal can be injected through
      this single component?

  REVERSE patch:
    Run the ATTACK input but replace one layer-component's hidden state
    with the corresponding hidden state from the CONTROL run.
    → Measures: how much of the attack signal is destroyed when we
      remove this component's contribution?

Both effects are normalised to [0, 1] by dividing by:
    attack_score - control_score

See src/patching.py for full documentation of the computation.

The patching batch size is 1 (one forward pass at a time) for clarity.
Each forward pass installs one hook, runs the model, and removes the hook.
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys
from typing import Dict, List

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import yaml

from src.model_utils import (
    get_true_false_token_ids,
    load_monot5,
    resolve_device,
)
from src.patching import aggregate_results, run_layer_patching_for_example


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stage 03: run layer-level activation patching."
    )
    p.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "default.yaml"),
        help="Path to the YAML config file.",
    )
    p.add_argument(
        "--attack-name",
        default=None,
        help="If set, read/write outputs under outputs/attacks/{attack-name}/.",
    )
    return p.parse_args()


def _resolve_base_dir(cfg: dict, attack_name: str | None) -> pathlib.Path:
    base = PROJECT_ROOT / cfg["outputs"]["base_dir"]
    if attack_name:
        return base / "attacks" / attack_name
    return base


def main() -> None:
    args = parse_args()
    cfg_path = pathlib.Path(args.config).resolve()
    print(f"[03_run_layer_patching] Loading config from: {cfg_path}")

    with open(cfg_path) as fh:
        cfg = yaml.safe_load(fh)

    model_cfg   = cfg["model"]
    attack_cfg  = cfg["attack"]
    runtime_cfg = cfg["runtime"]

    base_dir   = _resolve_base_dir(cfg, args.attack_name)
    selected_path  = base_dir / "scores" / "selected_examples.jsonl"
    act_dir    = base_dir / "activations"
    patch_dir  = base_dir / "patching"
    patch_dir.mkdir(parents=True, exist_ok=True)

    if not selected_path.exists():
        sys.exit(
            f"ERROR: {selected_path} not found.\n"
            "Please run Stage 01 first."
        )

    max_cached     = runtime_cfg["max_cached_examples"]

    # Load selected examples
    examples: List[Dict] = []
    with open(selected_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    print(f"[03_run_layer_patching] Loaded {len(examples)} selected examples.")

    # Enforce max_cached_examples (must match Stage 02)
    if len(examples) > max_cached:
        examples = examples[:max_cached]
        print(f"[03_run_layer_patching] Truncated to {max_cached} (max_cached_examples).")

    # Load model
    device = resolve_device(model_cfg["device"])
    model, tokenizer = load_monot5(model_cfg["checkpoint"], device)
    true_id, false_id = get_true_false_token_ids(tokenizer)

    all_detailed: List[Dict] = []
    skipped = 0

    for i, ex in enumerate(examples):
        qid   = ex["qid"]
        docid = ex["docid"]
        print(
            f"[03_run_layer_patching] [{i+1}/{len(examples)}] "
            f"qid={qid} docid={docid}  "
            f"delta={ex['attack_delta_vs_control']:.4f}"
        )

        # Load cached activations for this example
        control_cache_path = act_dir / f"{qid}_{docid}_control.pt"
        attack_cache_path  = act_dir / f"{qid}_{docid}_attack.pt"

        if not control_cache_path.exists() or not attack_cache_path.exists():
            print(
                f"  WARNING: cached activations not found for qid={qid} docid={docid}. "
                "Skipping. Did you run Stage 02?"
            )
            skipped += 1
            continue

        # Load activations and encodings from disk (both saved by Stage 02)
        control_data = torch.load(control_cache_path, map_location="cpu", weights_only=True)
        attack_data  = torch.load(attack_cache_path,  map_location="cpu", weights_only=True)
        control_cache = control_data["activations"]
        attack_cache  = attack_data["activations"]
        control_enc   = control_data["encoding"]  # {input_ids, attention_mask} on CPU
        attack_enc    = attack_data["encoding"]

        # Run all (layer, component, direction) patches for this example
        results = run_layer_patching_for_example(
            example=ex,
            model=model,
            control_enc=control_enc,
            attack_enc=attack_enc,
            control_cache=control_cache,
            attack_cache=attack_cache,
            true_id=true_id,
            false_id=false_id,
            device=device,
        )

        if results is None:
            # Example skipped due to negligible delta (logged inside the function)
            skipped += 1
            continue

        all_detailed.extend(results)
        print(f"  → {len(results)} patching results for this example.")

    print(
        f"\n[03_run_layer_patching] Processed {len(examples) - skipped} examples "
        f"({skipped} skipped)."
    )

    if not all_detailed:
        sys.exit(
            "ERROR: No patching results collected. "
            "Check that selected_examples and cached activations exist and are non-empty."
        )

    # -------------------------------------------------------------------------
    # Save detailed per-example results
    # -------------------------------------------------------------------------
    detailed_path = patch_dir / "layer_patching_results_detailed.csv"
    detailed_fields = [
        "qid", "docid", "layer", "component", "direction",
        "control_score", "attack_score", "patched_score", "effect",
    ]
    with open(detailed_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=detailed_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_detailed)
    print(f"[03_run_layer_patching] Saved detailed results → {detailed_path}")

    # -------------------------------------------------------------------------
    # Save aggregated results
    # -------------------------------------------------------------------------
    aggregated = aggregate_results(all_detailed)
    agg_path = patch_dir / "layer_patching_results.csv"
    agg_fields = [
        "layer", "component", "direction", "mean_effect", "std_effect", "n_examples"
    ]
    with open(agg_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=agg_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(aggregated)
    print(f"[03_run_layer_patching] Saved aggregated results → {agg_path}")
    print("[03_run_layer_patching] Done.")


if __name__ == "__main__":
    main()
