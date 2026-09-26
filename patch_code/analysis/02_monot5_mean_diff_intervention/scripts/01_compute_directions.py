#!/usr/bin/env python3
"""
scripts/01_compute_directions.py
===================================
Experiment 2, Part 1: compute mean-diff directions
(direction = mean(attack_activations) - mean(control_activations)) at
encoder_self_attn / decoder_self_attn / decoder_cross_attn, whole-vector and
per-head, for two tiers:

  grid_a  pooled over all 105 attacks, up to n_examples_per_attack each.
  grid_b  one canonical attack (relevant_start_5), up to n_examples.

The fitting pool is identical to Part 2's intervention test pool in both
tiers (same top-N examples from Experiment 1's selection) — see DECISIONS.md.

Outputs:
  outputs/directions/{tier}_directions.pt          (DirKey -> Tensor)
  outputs/directions/{tier}_direction_norms.csv     (one row per DirKey)
  outputs/directions/{tier}_status.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
import traceback

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
EXP_DIR = SCRIPT_DIR.parent
EXP1_DIR = EXP_DIR.parent / "01_monot5_layer_patching"
EXP3_DIR = EXP_DIR.parent / "03_monot5_head_patching_ablation"
sys.path.insert(0, str(EXP1_DIR))
sys.path.insert(0, str(EXP3_DIR))
sys.path.insert(0, str(EXP_DIR))

from src.model_utils import (  # noqa: E402
    build_padded_control_and_attack_encodings_general,
    get_true_false_token_ids,
    load_monot5,
    resolve_device,
)
from src.patching import SKIP_EPSILON  # noqa: E402

from exp2lib.direction_fit import new_accumulator  # noqa: E402
from exp2lib.direction_hooks import cache_activations_for_direction  # noqa: E402
from exp2lib.run_utils import (  # noqa: E402
    attacks_for_tier,
    load_config,
    load_reused_examples,
    n_examples_for_tier,
    resolve_cfg_path,
    save_directions,
    save_norm_rows,
)

TIER_NAMES = ["grid_a", "grid_b"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Part 1: mean-diff direction computation.")
    p.add_argument("--config", default=str(EXP_DIR / "configs" / "default.yaml"))
    p.add_argument("--tier", choices=TIER_NAMES + ["all"], default="all")
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def process_tier(tier: str, cfg: dict, outputs_dir: pathlib.Path,
                  model, tokenizer, true_id: int, false_id: int, device) -> dict:
    max_length = cfg["model"]["max_length"]
    n_encoder_layers = model.config.num_layers
    n_decoder_layers = model.config.num_decoder_layers
    reuse_dir = resolve_cfg_path(cfg, cfg["selection"]["reuse_dir"])
    n_examples = n_examples_for_tier(tier, cfg)

    acc = new_accumulator(model)
    status = {
        "tier": tier, "status": "failed", "error": None,
        "n_attacks": 0, "n_examples_used": 0,
        "n_align_failed": 0, "n_skipped_epsilon": 0,
    }
    t0 = time.time()
    try:
        specs = attacks_for_tier(tier, cfg)
        status["n_attacks"] = len(specs)
        for spec in specs:
            examples = load_reused_examples(spec.attack_name, n_examples, reuse_dir)
            if examples is None:
                print(f"    [WARN] no selection file for {spec.attack_name} — skipping.")
                continue
            for ex in examples:
                control_enc, attack_enc, align_result = (
                    build_padded_control_and_attack_encodings_general(
                        tokenizer=tokenizer, query=ex["query"], passage=ex["passage"],
                        attacked_passage=ex["attacked_passage"], max_length=max_length,
                        device=device,
                    )
                )
                if align_result.status != "ok":
                    status["n_align_failed"] += 1
                    continue

                ctrl_whole, ctrl_head, control_score = cache_activations_for_direction(
                    model, control_enc, true_id, false_id, device,
                    n_encoder_layers, n_decoder_layers,
                )
                atk_whole, atk_head, attack_score = cache_activations_for_direction(
                    model, attack_enc, true_id, false_id, device,
                    n_encoder_layers, n_decoder_layers,
                )
                if abs(attack_score - control_score) < SKIP_EPSILON:
                    status["n_skipped_epsilon"] += 1
                    continue

                acc.add_example("control", ctrl_whole, ctrl_head, control_enc["attention_mask"])
                acc.add_example("attack", atk_whole, atk_head, attack_enc["attention_mask"])
                status["n_examples_used"] += 1
            elapsed = time.time() - t0
            print(f"    [{tier}] {spec.attack_name}: "
                  f"{status['n_examples_used']} examples so far ({elapsed:.0f}s elapsed)",
                  flush=True)

        directions = acc.compute_directions()
        norm_rows = acc.norm_rows(directions, tier)
        save_directions(directions, outputs_dir / f"{tier}_directions.pt")
        save_norm_rows(norm_rows, outputs_dir / f"{tier}_direction_norms.csv")
        status["status"] = "success"
        status["n_direction_keys"] = len(directions)
    except Exception:
        tb = traceback.format_exc()
        print(f"\nERROR in tier {tier}:\n{tb}")
        status["error"] = tb

    with open(outputs_dir / f"{tier}_status.json", "w", encoding="utf-8") as fh:
        json.dump(status, fh, indent=2)
    return status


def main() -> None:
    args = parse_args()
    cfg = load_config(pathlib.Path(args.config))
    tiers = TIER_NAMES if args.tier == "all" else [args.tier]
    outputs_base = resolve_cfg_path(cfg, cfg["outputs"]["base_dir"])
    directions_dir = outputs_base / "directions"
    directions_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(cfg["model"]["device"])
    model, tokenizer = load_monot5(cfg["model"]["checkpoint"], device)
    true_id, false_id = get_true_false_token_ids(tokenizer)

    statuses = []
    for tier in tiers:
        status_path = directions_dir / f"{tier}_status.json"
        if not args.force and status_path.exists():
            with open(status_path, encoding="utf-8") as fh:
                prev = json.load(fh)
            if prev.get("status") == "success":
                print(f"[RESUME] tier {tier} already done — skipping.")
                statuses.append(prev)
                continue
        print(f"\n{'='*60}\n  TIER {tier}\n{'='*60}")
        statuses.append(process_tier(
            tier, cfg, directions_dir, model, tokenizer, true_id, false_id, device
        ))

    print(f"\n{'='*60}\n  SUMMARY\n{'='*60}")
    for s in statuses:
        icon = "✓" if s["status"] == "success" else "✗"
        print(f"  {icon} {s['tier']:8s} status={s['status']:8s} "
              f"examples={s.get('n_examples_used', 0):5d} "
              f"keys={s.get('n_direction_keys', 0)}")
    n_failed = sum(1 for s in statuses if s["status"] != "success")
    if n_failed:
        sys.exit(f"\n{n_failed} tier(s) failed — re-run to resume.")
    print(f"\n  Next: python scripts/02_run_interventions.py --config {args.config}")


if __name__ == "__main__":
    main()
