#!/usr/bin/env python3
"""
scripts/01_score_and_select.py
===============================
Stage 01: Score all pairs with monoT5 and select examples where the attack works.

Input
-----
  outputs/pairs/pairs.jsonl      (produced by Stage 00)

Outputs
-------
  outputs/scores/all_scores.csv         – all pairs with all three scores
  outputs/scores/selected_examples.jsonl – subset where attack > control

What we score
-------------
For each (query, passage) pair we compute three monoT5 scores:

  original_score  : "Query: Q Document: P Relevant:"
                    (original clean input — no prefix)

  control_score   : padded control — attack-prefix positions are filled with
                    tokenizer.pad_token_id, attention_mask=0 for those tokens.
                    This ensures passage tokens sit at identical absolute
                    positions in both sequences while the attack-prefix slots
                    contribute nothing to the model computation.

  attack_score    : "Query: Q Document: relevant:×5 P Relevant:"
                    (adversarial keyword-stuffing prefix from the paper)

We then compute:
  control_delta_vs_original = control_score - original_score
  attack_delta_vs_control   = attack_score  - control_score
  attack_delta_vs_original  = attack_score  - original_score

Selection criterion:
  attack_delta_vs_control > min_attack_delta

See src/scoring.py and src/model_utils.py for detailed explanations of
the scoring formula and why logit("true") - logit("false") is used.
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

import yaml

from src.data_utils import load_pairs
from src.model_utils import (
    get_true_false_token_ids,
    load_monot5,
    resolve_device,
)
from src.scoring import score_all_variants, select_attacked_examples


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stage 01: score all pairs and select attacked examples."
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
    print(f"[01_score_and_select] Loading config from: {cfg_path}")

    with open(cfg_path) as fh:
        cfg = yaml.safe_load(fh)

    model_cfg    = cfg["model"]
    attack_cfg   = cfg["attack"]
    selection_cfg = cfg["selection"]
    runtime_cfg  = cfg["runtime"]

    base_dir = _resolve_base_dir(cfg, args.attack_name)
    pairs_path = base_dir / "pairs" / "pairs.jsonl"
    scores_dir = base_dir / "scores"
    scores_dir.mkdir(parents=True, exist_ok=True)

    if not pairs_path.exists():
        sys.exit(
            f"ERROR: pairs file not found at {pairs_path}.\n"
            "Please run Stage 00 first:  python scripts/00_prepare_pairs.py"
        )

    # Load data
    pairs = load_pairs(pairs_path)
    print(f"[01_score_and_select] Loaded {len(pairs)} pairs.")

    # Load model
    device = resolve_device(model_cfg["device"])
    model, tokenizer = load_monot5(model_cfg["checkpoint"], device)
    true_id, false_id = get_true_false_token_ids(tokenizer)
    print(f"[01_score_and_select] true_id={true_id}, false_id={false_id}")

    # Score all three variants for every pair
    scored = score_all_variants(
        pairs=pairs,
        model=model,
        tokenizer=tokenizer,
        true_id=true_id,
        false_id=false_id,
        max_length=model_cfg["max_length"],
        device=device,
        batch_size=runtime_cfg["batch_size_scoring"],
    )

    # Save all scores
    all_scores_path = scores_dir / "all_scores.csv"
    fieldnames = [
        "qid", "docid", "rank", "bm25_score",
        "original_score", "control_score", "attack_score",
        "control_delta_vs_original", "attack_delta_vs_control", "attack_delta_vs_original",
    ]
    with open(all_scores_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(scored)
    print(f"[01_score_and_select] Saved all scores → {all_scores_path}")

    # Select examples where the attack was effective
    selected = select_attacked_examples(
        scored_pairs=scored,
        min_attack_delta=selection_cfg["min_attack_delta"],
        max_selected=selection_cfg["max_selected_examples"],
        seed=runtime_cfg["seed"],
    )

    # Save selected examples as JSONL
    selected_path = scores_dir / "selected_examples.jsonl"
    with open(selected_path, "w", encoding="utf-8") as fh:
        for ex in selected:
            fh.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(
        f"[01_score_and_select] Saved {len(selected)} selected examples → {selected_path}"
    )

    if len(selected) == 0:
        print(
            "\nWARNING: No examples selected. This may mean:\n"
            "  - The attack has no effect on this set of pairs.\n"
            "  - min_attack_delta is set too high.\n"
            "  - The padded control and attacked input produce very similar scores.\n"
            "Try setting min_attack_delta: -999 in the config to keep all pairs."
        )


if __name__ == "__main__":
    main()
