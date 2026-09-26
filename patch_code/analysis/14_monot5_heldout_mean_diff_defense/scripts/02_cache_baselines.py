#!/usr/bin/env python3
"""
scripts/02_cache_baselines.py
================================
Consolidate, per split, the successful-instance and clean-instance example
pools every later stage needs — reusing Experiment 1's already-computed
scores (control_score, attack_score, original_score) rather than rescoring
(task spec section 12: "Cache ... do not recompute them once cached" — here
that means not re-reading+re-filtering 105 selected_examples.jsonl files
once per downstream script).

No model forward passes happen in this script — it is pure I/O/filtering
over Experiment 1's cached scores (see exp14lib/data_pool.py).

Outputs (per split):
  outputs/baselines/{split}_by_attack.json      {attack_name: [examples]}
  outputs/baselines/{split}_clean_examples.json [ {qid, docid, query, passage, original_score} ]
  outputs/baselines/train_pooled_examples.json  round-robin-pooled TRAIN examples, capped
  outputs/baselines/status.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
EXP_DIR = SCRIPT_DIR.parent
EXP1_DIR = EXP_DIR.parent / "01_monot5_layer_patching"
EXP3_DIR = EXP_DIR.parent / "03_monot5_head_patching_ablation"
sys.path.insert(0, str(EXP1_DIR))
sys.path.insert(0, str(EXP3_DIR))
sys.path.insert(0, str(EXP_DIR))

from exp14lib.attacks import list_all_attacks  # noqa: E402
from exp14lib.data_pool import (  # noqa: E402
    collect_clean_examples_for_split,
    load_attack_examples,
    pool_round_robin,
)
from exp14lib.run_utils import (  # noqa: E402
    is_already_successful,
    load_config,
    resolve_cfg_path,
    write_status,
)
from exp14lib.splits import load_split_manifest, pair_to_split_map  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cache split-aware successful/clean example pools.")
    p.add_argument("--config", default=str(EXP_DIR / "configs" / "default.yaml"))
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(pathlib.Path(args.config))
    outputs_base = resolve_cfg_path(cfg, cfg["outputs"]["base_dir"])
    baselines_dir = outputs_base / "baselines"

    required = ["train_by_attack.json", "validation_by_attack.json", "test_by_attack.json",
                "train_clean_examples.json", "validation_clean_examples.json", "test_clean_examples.json",
                "train_pooled_examples.json"]
    if not args.force and is_already_successful(baselines_dir, required):
        print("[02_cache_baselines] RESUME: already done — skipping.")
        return

    split_manifest_path = outputs_base / "split_manifest.json"
    splits = load_split_manifest(split_manifest_path)
    pair_split = pair_to_split_map(splits)

    exp1_attacks_dir = resolve_cfg_path(cfg, cfg["data"]["exp1_attacks_dir"])
    all_specs = list_all_attacks(cfg)
    attack_names = [s.attack_name for s in all_specs]

    sampling = cfg["sampling"]
    per_split_example_cap = {
        "train": None,  # capped later via round-robin pooling, not per-attack here
        "validation": sampling["max_validation_examples_per_condition"],
        "test": sampling["max_test_examples_per_condition"],
    }
    per_split_clean_cap = {
        "train": sampling["max_clean_pairs_per_split"],
        "validation": sampling["max_clean_pairs_per_split"],
        "test": sampling["max_clean_pairs_per_split"],
    }

    baselines_dir.mkdir(parents=True, exist_ok=True)
    status = {"status": "failed", "splits": {}}
    try:
        for split_name in ("train", "validation", "test"):
            by_attack = {}
            n_total = 0
            for attack_name in attack_names:
                examples = load_attack_examples(
                    attack_name, exp1_attacks_dir, pair_split, split_name,
                    cap=per_split_example_cap[split_name],
                )
                if examples:
                    by_attack[attack_name] = examples
                    n_total += len(examples)
            with open(baselines_dir / f"{split_name}_by_attack.json", "w", encoding="utf-8") as fh:
                json.dump(by_attack, fh)

            clean_examples = collect_clean_examples_for_split(
                attack_names, exp1_attacks_dir, pair_split, split_name,
                cap=per_split_clean_cap[split_name],
            )
            with open(baselines_dir / f"{split_name}_clean_examples.json", "w", encoding="utf-8") as fh:
                json.dump(clean_examples, fh)

            status["splits"][split_name] = {
                "n_attacks_with_examples": len(by_attack),
                "n_examples_total": n_total,
                "n_clean_examples": len(clean_examples),
            }
            print(f"[02_cache_baselines] {split_name}: {len(by_attack)}/{len(attack_names)} attacks have "
                  f"successful instances, {n_total} examples total, {len(clean_examples)} clean pairs.")

        # Round-robin pooled TRAIN set for direction fitting (03_fit_directions.py).
        with open(baselines_dir / "train_by_attack.json", encoding="utf-8") as fh:
            train_by_attack = json.load(fh)
        pooled = pool_round_robin(train_by_attack, cap=sampling["max_train_examples_total"])
        with open(baselines_dir / "train_pooled_examples.json", "w", encoding="utf-8") as fh:
            json.dump(pooled, fh)
        status["n_train_pooled_examples"] = len(pooled)
        print(f"[02_cache_baselines] train_pooled_examples: {len(pooled)} "
              f"(round-robin across {len(train_by_attack)} attacks, cap={sampling['max_train_examples_total']})")

        status["status"] = "success"
    finally:
        write_status(baselines_dir, status)

    if status["status"] != "success":
        sys.exit(1)


if __name__ == "__main__":
    main()
