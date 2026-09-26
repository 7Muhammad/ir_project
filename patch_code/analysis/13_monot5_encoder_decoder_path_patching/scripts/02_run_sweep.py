#!/usr/bin/env python3
"""
scripts/02_run_sweep.py
=========================
Main 558-path sweep over all (attack, example) pairs. Writes one raw CSV
per attack (outputs/raw/{attack_name}/paths.csv, one row per
sender x receiver) plus a per-attack status.json used for checkpoint/resume:
an example already marked "completed" or "skipped" is never recomputed on
a re-run of this script, so it can be safely killed and restarted.

Usage:
    python scripts/02_run_sweep.py --config configs/default.yaml
    python scripts/02_run_sweep.py --config configs/default.yaml --max-examples 30
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys
import time

EXP_DIR = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(EXP_DIR))

from exp13lib.head_lists import assert_head_sets, load_receivers, load_senders  # noqa: E402
from exp13lib.path_engine import run_example_paths  # noqa: E402
from exp13lib.run_utils import (  # noqa: E402
    build_encodings,
    get_attacks,
    load_config,
    load_model,
    resolve_cfg_path,
    select_examples_for_attack,
)

ROW_FIELDS = [
    "attack_name", "qid", "docid", "sample_index", "seed",
    "sender_layer", "sender_head", "sender_name",
    "receiver_layer", "receiver_head", "receiver_name",
    "score_control", "score_attack", "delta",
    "score_path_forward", "score_path_reverse",
    "path_forward", "path_reverse", "path_combined",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Experiment 13 -- main sweep with checkpoint/resume.")
    p.add_argument("--config", default=str(EXP_DIR / "configs" / "default.yaml"))
    p.add_argument("--max-examples", type=int, default=None, help="Override run.max_examples_per_attack.")
    p.add_argument("--attacks", nargs="*", default=None, help="Override run.attack_subset with these attack names.")
    return p.parse_args()


def load_status(status_path: pathlib.Path) -> dict:
    if status_path.exists():
        with open(status_path, encoding="utf-8") as fh:
            return json.load(fh)
    return {"completed_examples": [], "skipped_examples": {}}


def save_status(status_path: pathlib.Path, status: dict) -> None:
    tmp = status_path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(status, fh, indent=2)
    tmp.replace(status_path)


def main() -> None:
    args = parse_args()
    cfg = load_config(pathlib.Path(args.config))

    max_examples = args.max_examples or cfg["run"]["max_examples_per_attack"]
    checkpoint_frequency = cfg["run"].get("checkpoint_frequency", 1)
    seed = cfg["runtime"]["seed"]

    senders = load_senders()
    receivers = load_receivers()
    assert_head_sets(senders, receivers)
    print(f"[sweep] {len(senders)} senders x {len(receivers)} receivers = {len(senders)*len(receivers)} paths/example")
    print(f"[sweep] max_examples_per_attack = {max_examples}, seed = {seed}")

    if args.attacks:
        cfg["run"]["attack_subset"] = args.attacks
    attacks = get_attacks(cfg)
    print(f"[sweep] {len(attacks)} attacks in scope")

    model, tokenizer, true_id, false_id, device = load_model(cfg)

    raw_dir = resolve_cfg_path(cfg, cfg["outputs"]["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    reuse_dir = resolve_cfg_path(cfg, cfg["selection"]["reuse_dir"])

    total_examples_done = 0
    t_start = time.time()

    for attack_idx, spec in enumerate(attacks, start=1):
        attack_dir = raw_dir / spec.attack_name
        attack_dir.mkdir(parents=True, exist_ok=True)
        csv_path = attack_dir / "paths.csv"
        status_path = attack_dir / "status.json"
        status = load_status(status_path)

        examples = select_examples_for_attack(spec.attack_name, reuse_dir, max_examples, seed)
        if len(examples) > max_examples:
            raise AssertionError(f"{spec.attack_name}: selection returned {len(examples)} > cap {max_examples}")

        write_header = not csv_path.exists()
        csv_fh = open(csv_path, "a", newline="", encoding="utf-8")
        writer = csv.DictWriter(csv_fh, fieldnames=ROW_FIELDS)
        if write_header:
            writer.writeheader()

        n_done_this_attack = 0
        for example in examples:
            key = f"{example['qid']}::{example['docid']}"
            if key in status["completed_examples"] or key in status.get("skipped_examples", {}):
                continue

            control_enc, attack_enc, align_result = build_encodings(
                tokenizer, example, cfg["model"]["max_length"], device
            )
            if align_result.status != "ok":
                status.setdefault("skipped_examples", {})[key] = f"alignment_failed:{align_result.status}"
                save_status(status_path, status)
                continue

            meta = {
                "attack_name": spec.attack_name, "qid": example["qid"], "docid": example["docid"],
                "sample_index": example.get("sample_index"), "seed": example.get("seed", seed),
            }
            result = run_example_paths(
                model, control_enc, attack_enc, senders, receivers, true_id, false_id, device, meta,
            )
            if result is None:
                status.setdefault("skipped_examples", {})[key] = "delta_below_epsilon"
                save_status(status_path, status)
                continue

            for row in result["rows"]:
                writer.writerow(row)
            csv_fh.flush()

            status["completed_examples"].append(key)
            save_status(status_path, status)
            n_done_this_attack += 1
            total_examples_done += 1

            if total_examples_done % checkpoint_frequency == 0:
                elapsed = time.time() - t_start
                print(f"[sweep] {spec.attack_name}: {n_done_this_attack}/{len(examples)} examples "
                      f"({total_examples_done} total, {elapsed:.0f}s elapsed)")

        csv_fh.close()
        n_completed = len(status["completed_examples"])
        n_skipped = len(status.get("skipped_examples", {}))
        print(f"[sweep] [{attack_idx}/{len(attacks)}] {spec.attack_name}: "
              f"{n_completed} completed, {n_skipped} skipped (of {len(examples)} selected)")

    print(f"[sweep] done. {total_examples_done} examples processed in this run "
          f"({time.time()-t_start:.0f}s).")


if __name__ == "__main__":
    main()
