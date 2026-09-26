#!/usr/bin/env python3
"""
scripts/08_template_full_sweep.py
====================================
Experiment 6 extension, Step 4 -- full 105-attack sweep of the Step 1
single-position causal patch (all 7 template positions, no pre-filtering
by the canonical-attack result, since attack success varies strongly by
token/position/repetition -- see src/attack_registry.py and the main
report's §4.1).

Resume-safe per attack (status.json + results.csv), same convention as
scripts/02_run_grid.py: skip an attack already marked "success" unless
--force. n_examples per attack from configs/template_tokens.yaml's
runs.sweep.n_examples (10-20, matching the existing §4.4 full-grid
convention of n=10/attack).
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
sys.path.insert(0, str(EXP1_DIR))
sys.path.insert(0, str(EXP_DIR))

from src.attack_registry import AttackSpec, discover_attacks
from src.model_utils import get_true_false_token_ids, load_monot5, resolve_device

from exp6lib.run_utils import build_example_inputs, get_examples_for_attack, load_config, resolve_cfg_path
from exp6lib.template_engine import run_template_position_causal_patch_example
from exp6lib.template_positions import get_template_positions

FIELDS = [
    "qid", "docid", "attack_name", "region", "layer", "template_position", "n_tokens",
    "score_control", "score_attack", "score_patched_fwd", "score_patched_rev",
    "fwd_effect", "rev_effect", "combined_effect",
    "query_span_start", "query_span_end", "doc_span_start", "doc_span_end",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Experiment 6 extension -- Step 4 full 105-attack sweep.")
    p.add_argument("--config", default=str(EXP_DIR / "configs" / "template_tokens.yaml"))
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
    return st.get("status") == "success" and (attack_dir / "results.csv").exists()


def process_attack(spec: AttackSpec, cfg: dict, out_dir: pathlib.Path, model, tokenizer,
                    true_id, false_id, device, encoder_layers: List[int]) -> dict:
    n_examples = cfg["runs"]["sweep"]["n_examples"]
    max_length = cfg["model"]["max_length"]

    out_dir.mkdir(parents=True, exist_ok=True)
    status = {
        "attack_name": spec.attack_name, "status": "failed", "error": None,
        "n_examples_requested": n_examples, "n_examples_used": 0,
        "n_align_failed": 0, "n_skipped_epsilon": 0, "n_rows": 0,
    }
    _write_status(out_dir, status)

    try:
        examples = get_examples_for_attack(spec, n_examples, cfg, model, tokenizer, true_id, false_id, device)
        with open(out_dir / "selected_examples.jsonl", "w", encoding="utf-8") as fh:
            for ex in examples:
                fh.write(json.dumps(ex, ensure_ascii=False) + "\n")

        csv_path = out_dir / "results.csv"
        t0 = time.time()
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
            writer.writeheader()

            for i, ex in enumerate(examples):
                inputs = build_example_inputs(tokenizer, ex, max_length, device)
                if inputs.status != "ok":
                    status["n_align_failed"] += 1
                    continue

                attack_ids = inputs.attack_enc["input_ids"][0].tolist()
                template_positions = get_template_positions(
                    tokenizer, attack_ids, inputs.query_span, inputs.doc_span_control_attack,
                )
                rows = run_template_position_causal_patch_example(
                    model, inputs.control_enc, inputs.attack_enc, encoder_layers,
                    template_positions, true_id, false_id, device,
                )
                if not rows:
                    status["n_skipped_epsilon"] += 1
                    continue

                for row in rows:
                    row["qid"] = ex["qid"]
                    row["docid"] = ex["docid"]
                    row["attack_name"] = spec.attack_name
                    row["query_span_start"], row["query_span_end"] = inputs.query_span
                    row["doc_span_start"], row["doc_span_end"] = inputs.doc_span_control_attack

                writer.writerows(rows)
                fh.flush()
                status["n_examples_used"] += 1
                status["n_rows"] += len(rows)

            elapsed = time.time() - t0
            print(f"    [{spec.attack_name}] {status['n_examples_used']}/{len(examples)} examples, "
                  f"{status['n_rows']} rows ({elapsed/max(1,len(examples)):.1f}s/ex)", flush=True)

        status["status"] = "success"

    except Exception:
        tb = traceback.format_exc()
        print(f"\nERROR in {spec.attack_name}:\n{tb}")
        status["error"] = tb

    _write_status(out_dir, status)
    return status


def main() -> None:
    args = parse_args()
    cfg = load_config(pathlib.Path(args.config))
    outputs_base = resolve_cfg_path(cfg, cfg["outputs"]["base_dir"])
    sweep_dir = outputs_base / "full_sweep"
    print(f"[08_template_full_sweep] Config:  {args.config}")
    print(f"[08_template_full_sweep] Outputs: {sweep_dir}")

    device = resolve_device(cfg["model"]["device"])
    model, tokenizer = load_monot5(cfg["model"]["checkpoint"], device)
    true_id, false_id = get_true_false_token_ids(tokenizer)

    encoder_layers = list(range(model.config.num_layers))
    specs = discover_attacks(cfg["attacks"])
    print(f"[08_template_full_sweep] {len(specs)} attack(s), "
          f"n_examples={cfg['runs']['sweep']['n_examples']}/attack, "
          f"{len(encoder_layers)} layers x 7 positions (encoder_self_attn only)")

    all_statuses = []
    for spec in specs:
        out_dir = sweep_dir / "attacks" / spec.attack_name
        if not args.force and _is_already_successful(out_dir):
            print(f"  [RESUME] {spec.attack_name} already done -- skipping.")
            with open(out_dir / "status.json", encoding="utf-8") as fh:
                all_statuses.append(json.load(fh))
            continue
        print(f"  [START ] {spec.attack_name}")
        all_statuses.append(process_attack(
            spec, cfg, out_dir, model, tokenizer, true_id, false_id, device, encoder_layers,
        ))

    print(f"\n{'='*60}\n  SUMMARY\n{'='*60}")
    for s in all_statuses:
        icon = "OK" if s["status"] == "success" else "FAIL"
        print(f"  [{icon:4s}] {s['attack_name']:30s} examples={s['n_examples_used']:4d} "
              f"rows={s['n_rows']:7d} align_failed={s['n_align_failed']}")
    n_failed = sum(1 for s in all_statuses if s["status"] != "success")
    if n_failed:
        sys.exit(f"\n{n_failed} attack(s) failed -- re-run to resume.")
    print(f"\n  Next: python scripts/09_template_aggregate_and_plot.py --config {args.config}")


if __name__ == "__main__":
    main()
