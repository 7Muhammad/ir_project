#!/usr/bin/env python3
"""
scripts/02_run_interventions.py
==================================
Experiment 2, Part 2: intervention tests at the 22 Experiment-3-flagged
decoder heads (outputs/flagged_heads.json, from script 00).

  2a defense (subtract, priority): base = attacked input.
  2b sufficiency (add): base = clean input.

Both run at every flagged head x scale in {0.5, 1.0, 1.5}, on the SAME
example pool used to fit that tier's direction in script 01 (grid_a: 105
attacks x up to 10 examples; grid_b: relevant_start_5 x up to 100 examples).

Per-example, per-row output (not pre-aggregated):
  outputs/interventions/{tier}/attacks/{attack_name}/rows.csv
  outputs/interventions/{tier}/attacks/{attack_name}/status.json

Resume-safe: attacks whose status.json says "success" are skipped unless
--force is given (same pattern as scripts 01/10 of Experiments 1/3).
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys
import time
import traceback
from typing import List

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
EXP_DIR = SCRIPT_DIR.parent
EXP1_DIR = EXP_DIR.parent / "01_monot5_layer_patching"
EXP3_DIR = EXP_DIR.parent / "03_monot5_head_patching_ablation"
sys.path.insert(0, str(EXP1_DIR))
sys.path.insert(0, str(EXP3_DIR))
sys.path.insert(0, str(EXP_DIR))

from src.attack_registry import AttackSpec  # noqa: E402
from src.model_utils import (  # noqa: E402
    build_padded_control_and_attack_encodings_general,
    get_true_false_token_ids,
    load_monot5,
    resolve_device,
    score_from_encoding,
)
from src.patching import SKIP_EPSILON  # noqa: E402

from exp2lib.intervene import run_defense_example, run_sufficiency_example  # noqa: E402
from exp2lib.run_utils import (  # noqa: E402
    attacks_for_tier,
    build_clean_encoding,
    load_config,
    load_direction_norms,
    load_directions,
    load_reused_examples,
    n_examples_for_tier,
    resolve_cfg_path,
)

TIER_NAMES = ["grid_a", "grid_b"]

ROW_FIELDS = [
    "qid", "docid", "attack_name", "layer", "component", "head_idx",
    "granularity", "scale", "intervention_type",
    "score_clean", "score_control", "score_attack", "score_modified",
    "delta_toward_control", "delta_toward_attack", "direction_norm",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Part 2: intervention tests.")
    p.add_argument("--config", default=str(EXP_DIR / "configs" / "default.yaml"))
    p.add_argument("--tier", choices=TIER_NAMES + ["all"], default="all")
    p.add_argument("--intervention", choices=["defense", "sufficiency", "both"], default="both")
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def _write_status(attack_dir: pathlib.Path, status: dict) -> None:
    with open(attack_dir / "status.json", "w", encoding="utf-8") as fh:
        json.dump(status, fh, indent=2)


def _is_already_successful(attack_dir: pathlib.Path) -> bool:
    status_path = attack_dir / "status.json"
    if not status_path.exists():
        return False
    try:
        with open(status_path, encoding="utf-8") as fh:
            st = json.load(fh)
    except Exception:
        return False
    return st.get("status") == "success" and (attack_dir / "rows.csv").exists()


def process_attack(
    tier: str, spec: AttackSpec, cfg: dict, out_dir: pathlib.Path,
    model, tokenizer, true_id: int, false_id: int, device,
    flagged_heads: List[dict], directions, direction_norms,
    scales: List[float], run_defense: bool, run_sufficiency: bool,
) -> dict:
    max_length = cfg["model"]["max_length"]
    reuse_dir = resolve_cfg_path(cfg, cfg["selection"]["reuse_dir"])
    n_examples = n_examples_for_tier(tier, cfg)

    out_dir.mkdir(parents=True, exist_ok=True)
    status = {
        "tier": tier, "attack_name": spec.attack_name, "status": "failed", "error": None,
        "n_examples_used": 0, "n_align_failed": 0, "n_skipped_epsilon": 0, "n_rows": 0,
    }
    _write_status(out_dir, status)

    try:
        examples = load_reused_examples(spec.attack_name, n_examples, reuse_dir)
        if examples is None:
            status["error"] = "no selection file"
            _write_status(out_dir, status)
            return status

        csv_path = out_dir / "rows.csv"
        t0 = time.time()
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=ROW_FIELDS, extrasaction="ignore")
            writer.writeheader()

            for i, ex in enumerate(examples):
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

                control_score = score_from_encoding(model, control_enc, true_id, false_id, device)
                attack_score = score_from_encoding(model, attack_enc, true_id, false_id, device)
                if abs(attack_score - control_score) < SKIP_EPSILON:
                    status["n_skipped_epsilon"] += 1
                    continue

                clean_enc = build_clean_encoding(
                    tokenizer, ex["query"], ex["passage"], max_length, device
                )
                clean_score = score_from_encoding(model, clean_enc, true_id, false_id, device)

                meta = {
                    "qid": ex["qid"], "docid": ex["docid"], "attack_name": spec.attack_name,
                    "score_clean": clean_score, "score_control": control_score,
                    "score_attack": attack_score,
                }

                rows = []
                if run_defense:
                    rows += run_defense_example(
                        model, attack_enc, flagged_heads, directions, direction_norms,
                        scales, true_id, false_id, device, meta,
                    )
                if run_sufficiency:
                    rows += run_sufficiency_example(
                        model, clean_enc, flagged_heads, directions, direction_norms,
                        scales, true_id, false_id, device, meta,
                    )

                writer.writerows(rows)
                fh.flush()
                status["n_examples_used"] += 1
                status["n_rows"] += len(rows)
                elapsed = time.time() - t0
                print(f"    [{tier}/{spec.attack_name}] example {i+1}/{len(examples)} "
                      f"qid={ex['qid']} docid={ex['docid']} rows={len(rows)} "
                      f"({elapsed/(i+1):.1f}s/ex)", flush=True)

        status["status"] = "success"
    except Exception:
        tb = traceback.format_exc()
        print(f"\nERROR in {tier}/{spec.attack_name}:\n{tb}")
        status["error"] = tb

    _write_status(out_dir, status)
    return status


def main() -> None:
    args = parse_args()
    cfg = load_config(pathlib.Path(args.config))
    tiers = TIER_NAMES if args.tier == "all" else [args.tier]
    run_defense = args.intervention in ("defense", "both")
    run_sufficiency = args.intervention in ("sufficiency", "both")
    scales = cfg["intervention"]["scales"]

    outputs_base = resolve_cfg_path(cfg, cfg["outputs"]["base_dir"])
    directions_dir = outputs_base / "directions"
    with open(outputs_base / "flagged_heads.json", encoding="utf-8") as fh:
        flagged_heads = json.load(fh)
    if not flagged_heads:
        sys.exit("flagged_heads.json is empty — run scripts/00_derive_flagged_heads.py first.")

    device = resolve_device(cfg["model"]["device"])
    model, tokenizer = load_monot5(cfg["model"]["checkpoint"], device)
    true_id, false_id = get_true_false_token_ids(tokenizer)

    all_statuses = []
    for tier in tiers:
        directions = load_directions(directions_dir / f"{tier}_directions.pt")
        direction_norms = load_direction_norms(directions_dir / f"{tier}_direction_norms.csv")

        specs = attacks_for_tier(tier, cfg)
        print(f"\n{'='*60}\n  TIER {tier}: {len(specs)} attack(s), "
              f"{len(flagged_heads)} flagged heads\n{'='*60}")
        for spec in specs:
            out_dir = outputs_base / "interventions" / tier / "attacks" / spec.attack_name
            if not args.force and _is_already_successful(out_dir):
                print(f"  [RESUME] {tier}/{spec.attack_name} already done — skipping.")
                with open(out_dir / "status.json", encoding="utf-8") as fh:
                    all_statuses.append(json.load(fh))
                continue
            print(f"  [START ] {tier}/{spec.attack_name}")
            all_statuses.append(process_attack(
                tier, spec, cfg, out_dir, model, tokenizer, true_id, false_id, device,
                flagged_heads, directions, direction_norms, scales,
                run_defense, run_sufficiency,
            ))

    print(f"\n{'='*60}\n  SUMMARY\n{'='*60}")
    for s in all_statuses:
        icon = "✓" if s["status"] == "success" else "✗"
        print(f"  {icon} {s['tier']:8s} {s['attack_name']:30s} status={s['status']:8s} "
              f"examples={s['n_examples_used']:4d} rows={s['n_rows']:7d}")
    n_failed = sum(1 for s in all_statuses if s["status"] != "success")
    if n_failed:
        sys.exit(f"\n{n_failed} attack unit(s) failed — re-run to resume.")
    print(f"\n  Next: python scripts/03_aggregate.py --config {args.config}")


if __name__ == "__main__":
    main()
