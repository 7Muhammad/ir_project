#!/usr/bin/env python3
"""
scripts/07_run_all_decoderlens_attacks.py
=========================================
End-to-end multi-attack orchestrator.

Loads monoT5 ONCE and runs the full per-attack DecoderLens pipeline for every
discovered attack:

    Stage 02  prepare target pairs (+ alignment)
    Stage 03  layerwise scoring (+ final-layer sanity check)
    Stage 04  layerwise ranks (+ summary)
    Stage 05  per-attack plots

Each attack writes its own outputs/attacks/{attack_name}/status.json.  If an
attack fails, the error is recorded and the runner continues to the next one.

After all attacks finish, the cross-attack comparison (Stage 06) is run unless
--no-compare is passed.

Status file schema (outputs/attacks/{attack_name}/status.json)
--------------------------------------------------------------
{
  "attack_name": "...", "token": "...", "position": "...",
  "repetitions": N, "run_name": "...",
  "status": "success" | "failed",
  "error": null | "<traceback>",
  "candidate_top_k": K,
  "n_alignment_success": ..., "n_alignment_failed": ...,
  "n_examples_used": ...,
  "sanity_passed": true/false, "sanity_max_abs_diff": ...
}

Usage
-----
  python scripts/07_run_all_decoderlens_attacks.py --config configs/default.yaml
  python scripts/07_run_all_decoderlens_attacks.py --config configs/default.yaml --force
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import sys
import traceback
from types import ModuleType
from typing import Dict

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.attack_registry import discover_attacks
from src.decoderlens import get_true_false_token_ids, load_monot5, resolve_device
from src.utils import attack_output_dirs, load_config


def _load_stage(filename: str) -> ModuleType:
    """Import a numbered stage script (whose name is not a valid identifier)."""
    path = SCRIPT_DIR / filename
    spec = importlib.util.spec_from_file_location(path.stem.replace("0", "s0"), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_status(base_dir: pathlib.Path, attack_name: str, status: Dict) -> None:
    out_dirs = attack_output_dirs(base_dir, attack_name)
    with open(out_dirs["root"] / "status.json", "w", encoding="utf-8") as fh:
        json.dump(status, fh, indent=2)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the full DecoderLens pipeline for all attacks.")
    p.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "default.yaml"))
    p.add_argument("--force", action="store_true",
                   help="Re-run attacks even if a successful status.json exists.")
    p.add_argument("--no-compare", action="store_true",
                   help="Skip the cross-attack comparison stage at the end.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    base_dir = PROJECT_ROOT / cfg["outputs"]["base_dir"]
    base_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(cfg["model"]["device"])
    model, tokenizer = load_monot5(cfg["model"]["checkpoint"], device)
    true_id, false_id = get_true_false_token_ids(tokenizer)

    stage02 = _load_stage("02_prepare_decoderlens_pairs.py")
    stage03 = _load_stage("03_run_layerwise_scoring.py")
    stage04 = _load_stage("04_compute_layerwise_ranks.py")
    stage05 = _load_stage("05_plot_per_attack_results.py")

    attacks = discover_attacks(cfg["attacks"])
    candidate_top_k = int(cfg["ranking"]["candidate_top_k"])
    print(f"[07] Running DecoderLens pipeline for {len(attacks)} attacks.")

    n_ok = n_failed = n_skipped = 0
    for i, spec in enumerate(attacks):
        out_dirs = attack_output_dirs(base_dir, spec.attack_name)
        status_path = out_dirs["root"] / "status.json"
        if status_path.exists() and not args.force:
            try:
                prev = json.loads(status_path.read_text())
                if prev.get("status") == "success":
                    print(f"[07] ({i+1}/{len(attacks)}) {spec.attack_name}: "
                          f"already done, skipping (use --force to rerun).")
                    n_skipped += 1
                    continue
            except json.JSONDecodeError:
                pass

        print(f"\n[07] ({i+1}/{len(attacks)}) === {spec.attack_name} ===")
        status: Dict = {
            "attack_name": spec.attack_name,
            "token": spec.token,
            "position": spec.position,
            "repetitions": spec.repetitions,
            "run_name": spec.run_name,
            "candidate_top_k": candidate_top_k,
            "status": "failed",
            "error": None,
        }
        try:
            report = stage02.prepare_pairs_for_attack(
                spec, cfg, tokenizer, device, base_dir
            )
            status["n_alignment_success"] = report["n_alignment_success"]
            status["n_alignment_failed"] = report["n_alignment_failed"]

            score_report = stage03.score_attack(
                spec, cfg, model, tokenizer, true_id, false_id, device, base_dir
            )
            status["n_examples_used"] = score_report.get("n_examples_used", 0)
            status["sanity_passed"] = score_report.get("sanity_passed")
            status["sanity_max_abs_diff"] = score_report.get("sanity_max_abs_diff")

            if status["n_examples_used"] and status["n_examples_used"] > 0:
                stage04.rank_attack(spec, cfg, base_dir)
                stage05.plot_attack(spec, cfg, base_dir)
            else:
                print(f"[07] {spec.attack_name}: no examples after selection; "
                      "skipping rank + plot.")

            status["status"] = "success"
            n_ok += 1
        except Exception:  # noqa: BLE001 - record and continue
            status["error"] = traceback.format_exc()
            n_failed += 1
            print(f"[07] ERROR in {spec.attack_name}:\n{status['error']}")

        _write_status(base_dir, spec.attack_name, status)

    print(f"\n[07] Done. success={n_ok} failed={n_failed} skipped={n_skipped}")

    if not args.no_compare:
        print("\n[07] Running cross-attack comparison (Stage 06) ...")
        stage06 = _load_stage("06_compare_decoderlens_attacks.py")
        sys.argv = ["06", "--config", args.config]
        stage06.main()


if __name__ == "__main__":
    main()
