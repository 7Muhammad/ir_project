#!/usr/bin/env python3
"""
scripts/02_run_grid.py
=========================
Experiment 7 full-grid driver: whole-block logit lens (all 12 decoder
layers) + per-head logit lens (Experiment-3-flagged heads), for clean,
control, and attack runs, across attacks in scope.

DO NOT RUN until scripts/01_run_pilot.py's extrapolated runtime has been
reviewed and configs/default.yaml's runs.grid.n_examples is confirmed.

Two run modes, matching Experiments 3/6's grid/canonical split:
  grid       every attack (up to attacks.max_attacks), runs.grid.n_examples each
  canonical  one named attack, runs.canonical.n_examples (deeper)

Resume-safe: attacks whose status.json says "success" and whose CSV exists
are skipped unless --force is given.
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

from exp7lib.engine import run_example
from exp7lib.run_utils import build_example_inputs, get_examples_for_attack, load_config, load_flagged_heads, resolve_cfg_path

RUN_NAMES = ["grid", "canonical"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Experiment 7 full-grid driver.")
    p.add_argument("--config", default=str(EXP_DIR / "configs" / "default.yaml"))
    p.add_argument("--run", choices=RUN_NAMES + ["all"], default="all")
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


def attacks_for_run(run_name: str, cfg: dict) -> List[AttackSpec]:
    if run_name == "grid":
        return discover_attacks(cfg["attacks"])
    canonical = cfg["runs"]["canonical"]["attack_name"]
    single_cfg = {
        "upstream_injected_dir": cfg["attacks"]["upstream_injected_dir"],
        "mode": "include", "include": [canonical],
    }
    return discover_attacks(single_cfg)


def process_attack(run_name, spec, cfg, out_dir, model, tokenizer, true_id, false_id, device, flagged_heads) -> dict:
    n_examples = cfg["runs"][run_name]["n_examples"]
    max_length = cfg["model"]["max_length"]
    k_values = cfg["k_values"]
    composition_k = cfg["composition_k"]

    out_dir.mkdir(parents=True, exist_ok=True)
    status = {
        "run": run_name, "attack_name": spec.attack_name, "status": "failed", "error": None,
        "n_examples_requested": n_examples, "n_examples_used": 0,
        "n_align_failed": 0, "n_rows": 0,
    }
    _write_status(out_dir, status)

    try:
        examples = get_examples_for_attack(spec, n_examples, cfg, model, tokenizer, true_id, false_id, device)
        with open(out_dir / "selected_examples.jsonl", "w", encoding="utf-8") as fh:
            for ex in examples:
                fh.write(json.dumps(ex, ensure_ascii=False) + "\n")

        csv_path = out_dir / "results.csv"
        fields = None
        t0 = time.time()
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            writer = None

            for i, ex in enumerate(examples):
                inputs = build_example_inputs(tokenizer, ex, spec.token, max_length, device)
                if inputs.status != "ok":
                    print(f"    ALIGN FAIL qid={ex['qid']} docid={ex['docid']}: {inputs.reason[:120]}")
                    status["n_align_failed"] += 1
                    continue

                rows = run_example(
                    model, tokenizer, inputs, ex["query"], ex["passage"], spec.token,
                    flagged_heads, k_values, composition_k, device,
                )
                for row in rows:
                    row["qid"], row["docid"], row["attack_name"] = ex["qid"], ex["docid"], spec.attack_name

                if writer is None:
                    fields = list(rows[0].keys())
                    writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
                    writer.writeheader()
                writer.writerows(rows)
                fh.flush()

                status["n_examples_used"] += 1
                status["n_rows"] += len(rows)
                elapsed = time.time() - t0
                print(f"    [{run_name}/{spec.attack_name}] example {i+1}/{len(examples)} "
                      f"qid={ex['qid']} docid={ex['docid']} rows={len(rows)} "
                      f"({elapsed/(i+1):.1f}s/ex)", flush=True)

        status["status"] = "success"

    except Exception:
        tb = traceback.format_exc()
        print(f"\nERROR in {run_name}/{spec.attack_name}:\n{tb}")
        status["error"] = tb

    _write_status(out_dir, status)
    return status


def main() -> None:
    args = parse_args()
    cfg = load_config(pathlib.Path(args.config))
    run_names = RUN_NAMES if args.run == "all" else [args.run]
    outputs_base = resolve_cfg_path(cfg, cfg["outputs"]["base_dir"])
    print(f"[02_run_grid] Config:  {args.config}")
    print(f"[02_run_grid] Runs:    {run_names}")
    print(f"[02_run_grid] Outputs: {outputs_base}")

    device = resolve_device(cfg["model"]["device"])
    model, tokenizer = load_monot5(cfg["model"]["checkpoint"], device)
    true_id, false_id = get_true_false_token_ids(tokenizer)

    flagged_heads = load_flagged_heads(cfg)
    print(f"[02_run_grid] Scope: {model.config.num_decoder_layers} whole-block layers + "
          f"{len(flagged_heads)} flagged heads, k={cfg['k_values']}")

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
                run_name, spec, cfg, out_dir, model, tokenizer, true_id, false_id, device, flagged_heads,
            ))

    print(f"\n{'='*60}\n  SUMMARY\n{'='*60}")
    for s in all_statuses:
        icon = "✓" if s["status"] == "success" else "✗"
        print(f"  {icon} {s['run']:10s} {s['attack_name']:30s} status={s['status']:8s} "
              f"examples={s['n_examples_used']:4d} rows={s['n_rows']:7d} align_failed={s['n_align_failed']}")
    n_failed = sum(1 for s in all_statuses if s["status"] != "success")
    if n_failed:
        sys.exit(f"\n{n_failed} attack unit(s) failed — re-run to resume.")
    print(f"\n  Next: python scripts/03_aggregate.py --config {args.config}")


if __name__ == "__main__":
    main()
