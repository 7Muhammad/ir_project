#!/usr/bin/env python3
"""
scripts/01_run_head_grid.py
===========================
Experiment 3 driver: per-head activation patching + ablation on the monoT5
decoder (12 layers x {self-attn, cross-attn} x 12 heads = 288 head-slots).

Four runs (select with --run, or --run all):

  grid_a   288 heads x 105 attacks x n_examples (default 10)
           fwd/rev/combined patching + zero/mean ablation on the ATTACK input.
  grid_b   288 heads x 1 canonical attack x n_examples (default 100),
           same metrics as grid_a.
  clean_a  zero/mean ablation while scoring grid_a's matched CLEAN inputs —
           side-effect check that ablation doesn't break normal scoring.
  clean_b  same for grid_b's clean inputs.

Outputs (per run, per attack) — PER-EXAMPLE rows, not aggregated:
  outputs/{run}/attacks/{attack_name}/head_results.csv
  outputs/{run}/attacks/{attack_name}/selected_examples.jsonl
  outputs/{run}/attacks/{attack_name}/status.json

Resume-safe like Experiment 1's script 10: attacks whose status.json says
"success" and whose CSV exists are skipped unless --force is given.

Aggregation and plots are separate:
  scripts/02_aggregate.py, scripts/03_make_plots.py
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys
import time
import traceback
from typing import Dict, List, Optional, Tuple

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
EXP_DIR = SCRIPT_DIR.parent                             # 03_monot5_head_patching_ablation
EXP1_DIR = EXP_DIR.parent / "01_monot5_layer_patching"  # sibling Experiment 1
sys.path.insert(0, str(EXP1_DIR))                 # for src.* (Experiment 1 code)
sys.path.insert(0, str(EXP_DIR))                  # for headlib.*

import torch

from src.attack_registry import AttackSpec, discover_attacks
from src.model_utils import (
    build_monot5_input,
    build_padded_control_and_attack_encodings_general,
    get_true_false_token_ids,
    load_monot5,
    resolve_device,
)

from headlib.engine import run_clean_example, run_grid_example
from headlib.head_hooks import enumerate_slots
from headlib.run_utils import get_examples_for_attack, load_config, resolve_cfg_path

RUN_NAMES = ["grid_a", "grid_b", "clean_a", "clean_b"]

GRID_FIELDS = [
    "qid", "docid", "attack_name", "layer", "component", "head_idx",
    "score_clean", "score_control", "score_attack",
    "score_patched_fwd", "score_patched_rev",
    "score_ablated_zero", "score_ablated_mean",
    "fwd_effect", "rev_effect", "combined_effect",
    "score_drop_zero", "score_drop_mean",
]

CLEAN_FIELDS = [
    "qid", "docid", "attack_name", "layer", "component", "head_idx",
    "score_clean", "score_control", "score_attack",
    "score_ablated_zero", "score_ablated_mean",
    "score_drop_zero", "score_drop_mean",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Per-head patching/ablation driver.")
    p.add_argument("--config", default=str(EXP_DIR / "configs" / "default.yaml"))
    p.add_argument("--run", choices=RUN_NAMES + ["all"], default="all")
    p.add_argument("--force", action="store_true",
                   help="Re-run attacks even if status.json says success.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Resume / status helpers (same pattern as Experiment 1's script 10)
# ---------------------------------------------------------------------------

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
    return st.get("status") == "success" and (attack_dir / "head_results.csv").exists()


# ---------------------------------------------------------------------------
# Attack resolution
# ---------------------------------------------------------------------------

def attacks_for_run(run_name: str, cfg: dict) -> List[AttackSpec]:
    """
    grid_a / clean_a sweep the inherited attack grid; grid_b / clean_b use the
    single canonical attack named in the config (resolved by name so it works
    even when the grid is truncated by max_attacks in smoke configs).
    """
    if run_name in ("grid_a", "clean_a"):
        return discover_attacks(cfg["attacks"])
    canonical = cfg["runs"][run_name]["attack_name"]
    single_cfg = {
        "upstream_injected_dir": cfg["attacks"]["upstream_injected_dir"],
        "mode": "include",
        "include": [canonical],
    }
    return discover_attacks(single_cfg)


# ---------------------------------------------------------------------------
# Per-attack processing
# ---------------------------------------------------------------------------

def process_attack(
    run_name: str,
    spec: AttackSpec,
    cfg: dict,
    out_dir: pathlib.Path,
    model,
    tokenizer,
    true_id: int,
    false_id: int,
    device: torch.device,
    slots: List[Tuple[str, int]],
) -> dict:
    """Run one (run, attack) unit and return its status dict."""
    is_clean_run = run_name.startswith("clean")
    n_examples = cfg["runs"][run_name]["n_examples"]
    methods = cfg["ablation"]["methods"]
    max_length = cfg["model"]["max_length"]

    out_dir.mkdir(parents=True, exist_ok=True)
    status = {
        "run": run_name,
        "attack_name": spec.attack_name,
        "status": "failed",
        "error": None,
        "n_examples_requested": n_examples,
        "n_examples_used": 0,
        "n_align_failed": 0,
        "n_skipped_epsilon": 0,
        "n_rows": 0,
    }
    _write_status(out_dir, status)

    try:
        examples = get_examples_for_attack(
            spec, n_examples, cfg, model, tokenizer, true_id, false_id, device
        )
        with open(out_dir / "selected_examples.jsonl", "w", encoding="utf-8") as fh:
            for ex in examples:
                fh.write(json.dumps(ex, ensure_ascii=False) + "\n")

        fields = CLEAN_FIELDS if is_clean_run else GRID_FIELDS
        csv_path = out_dir / "head_results.csv"
        t0 = time.time()
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()

            for i, ex in enumerate(examples):
                control_enc, attack_enc, align_result = (
                    build_padded_control_and_attack_encodings_general(
                        tokenizer=tokenizer,
                        query=ex["query"],
                        passage=ex["passage"],
                        attacked_passage=ex["attacked_passage"],
                        max_length=max_length,
                        device=device,
                    )
                )
                if align_result.status != "ok":
                    print(f"    ALIGN FAIL qid={ex['qid']} docid={ex['docid']}: "
                          f"{align_result.reason[:120]}")
                    status["n_align_failed"] += 1
                    continue

                meta = {
                    "qid": ex["qid"],
                    "docid": ex["docid"],
                    "attack_name": spec.attack_name,
                    "score_clean": ex.get("original_score"),
                    "score_control": ex.get("control_score"),
                    "score_attack": ex.get("attack_score"),
                }

                if is_clean_run:
                    clean_text = build_monot5_input(ex["query"], ex["passage"])
                    clean_tok = tokenizer(
                        clean_text, return_tensors="pt",
                        max_length=max_length, truncation=True, padding=False,
                    )
                    clean_enc = {
                        "input_ids": clean_tok["input_ids"],
                        "attention_mask": clean_tok["attention_mask"],
                    }
                    rows = run_clean_example(
                        model, clean_enc, control_enc, slots,
                        true_id, false_id, device, meta, methods,
                    )
                else:
                    rows = run_grid_example(
                        model, control_enc, attack_enc, slots,
                        true_id, false_id, device, meta, methods,
                    )
                    if rows is None:
                        status["n_skipped_epsilon"] += 1
                        continue

                writer.writerows(rows)
                fh.flush()
                status["n_examples_used"] += 1
                status["n_rows"] += len(rows)
                elapsed = time.time() - t0
                print(f"    [{run_name}/{spec.attack_name}] example "
                      f"{i+1}/{len(examples)} qid={ex['qid']} docid={ex['docid']} "
                      f"rows={len(rows)} ({elapsed/(i+1):.1f}s/ex)", flush=True)

        status["status"] = "success"

    except Exception:
        tb = traceback.format_exc()
        print(f"\nERROR in {run_name}/{spec.attack_name}:\n{tb}")
        status["error"] = tb

    _write_status(out_dir, status)
    return status


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    cfg = load_config(pathlib.Path(args.config))
    run_names = RUN_NAMES if args.run == "all" else [args.run]
    outputs_base = resolve_cfg_path(cfg, cfg["outputs"]["base_dir"])
    print(f"[01_run_head_grid] Config:  {args.config}")
    print(f"[01_run_head_grid] Runs:    {run_names}")
    print(f"[01_run_head_grid] Outputs: {outputs_base}")

    device = resolve_device(cfg["model"]["device"])
    model, tokenizer = load_monot5(cfg["model"]["checkpoint"], device)
    true_id, false_id = get_true_false_token_ids(tokenizer)

    heads_cfg = cfg.get("heads", {})
    slots = enumerate_slots(
        model,
        components=heads_cfg.get("components"),
        layers=heads_cfg.get("layers"),
    )
    n_heads = model.config.num_heads
    print(f"[01_run_head_grid] Scope: {len(slots)} (layer, component) slots "
          f"x {n_heads} heads = {len(slots) * n_heads} head-slots.")

    all_statuses: List[dict] = []
    for run_name in run_names:
        specs = attacks_for_run(run_name, cfg)
        print(f"\n{'='*60}\n  RUN {run_name}: {len(specs)} attack(s)\n{'='*60}")
        for spec in specs:
            out_dir = outputs_base / run_name / "attacks" / spec.attack_name
            if not args.force and _is_already_successful(out_dir):
                print(f"  [RESUME] {run_name}/{spec.attack_name} already done — skipping.")
                with open(out_dir / "status.json", encoding="utf-8") as fh:
                    all_statuses.append(json.load(fh))
                continue
            print(f"  [START ] {run_name}/{spec.attack_name}")
            all_statuses.append(process_attack(
                run_name, spec, cfg, out_dir,
                model, tokenizer, true_id, false_id, device, slots,
            ))

    print(f"\n{'='*60}\n  SUMMARY\n{'='*60}")
    for s in all_statuses:
        icon = "✓" if s["status"] == "success" else "✗"
        print(f"  {icon} {s['run']:8s} {s['attack_name']:30s} "
              f"status={s['status']:8s} examples={s['n_examples_used']:4d} "
              f"rows={s['n_rows']:7d} align_failed={s['n_align_failed']}")
    n_failed = sum(1 for s in all_statuses if s["status"] != "success")
    if n_failed:
        sys.exit(f"\n{n_failed} attack unit(s) failed — re-run to resume.")
    print(f"\n  Next: python scripts/02_aggregate.py --config {args.config}")


if __name__ == "__main__":
    main()
