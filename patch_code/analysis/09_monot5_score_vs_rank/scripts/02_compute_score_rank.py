#!/usr/bin/env python3
"""
scripts/02_compute_score_rank.py
===================================
Per attack, per target (up to 500, no filtering): compute delta_score
(from Experiment 1's already-scored all_scores.csv — zero forward passes)
and delta_rank (via DecoderLens's compute_rank against the fixed top-100
candidate set built by script 01).

Output: outputs/per_example/{attack_name}.csv
  attack_name, qid, docid, score_control, score_attack, delta_score,
  rank_control, rank_attack, delta_rank, success
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
EXP_DIR = SCRIPT_DIR.parent
EXP1_DIR = EXP_DIR.parent / "01_monot5_layer_patching"
EXP3_DIR = EXP_DIR.parent / "03_monot5_head_patching_ablation"
DECODERLENS_DIR = EXP_DIR.parent / "monot5_decoderlens_rank"
sys.path.insert(0, str(EXP1_DIR))
sys.path.insert(0, str(EXP3_DIR))
sys.path.insert(0, str(EXP_DIR))

from src.attack_registry import discover_attacks  # noqa: E402

from exp9lib.decoderlens_import import load_decoderlens_ranking  # noqa: E402
from exp9lib.metrics import success as success_fn  # noqa: E402
from exp9lib.run_utils import load_config, resolve_cfg_path  # noqa: E402
from exp9lib.targets import load_targets  # noqa: E402

ROW_FIELDS = [
    "attack_name", "qid", "docid", "score_control", "score_attack", "delta_score",
    "rank_control", "rank_attack", "delta_rank", "success",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute per-example score/rank deltas.")
    p.add_argument("--config", default=str(EXP_DIR / "configs" / "default.yaml"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(pathlib.Path(args.config))
    outputs_base = resolve_cfg_path(cfg, cfg["outputs"]["base_dir"])

    candidate_scores_path = outputs_base / "candidates" / "candidate_scores.json"
    if not candidate_scores_path.exists():
        sys.exit(f"{candidate_scores_path} not found — run scripts/01_build_candidate_scores.py first.")
    with open(candidate_scores_path, encoding="utf-8") as fh:
        candidate_scores = json.load(fh)

    _Candidate, _load_attack_tsv, _build_candidate_sets, compute_rank = (
        load_decoderlens_ranking(DECODERLENS_DIR)
    )

    exp1_outputs_dir = resolve_cfg_path(cfg, cfg["exp1"]["outputs_attacks_dir"])
    specs = discover_attacks(cfg["attacks"])
    per_example_dir = outputs_base / "per_example"
    per_example_dir.mkdir(parents=True, exist_ok=True)

    n_missing_qid = 0
    n_rows_total = 0
    for spec in specs:
        targets = load_targets(spec.attack_name, exp1_outputs_dir)
        out_path = per_example_dir / f"{spec.attack_name}.csv"
        with open(out_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=ROW_FIELDS)
            writer.writeheader()
            for t in targets:
                qid, docid = t["qid"], t["docid"]
                cand_scores = candidate_scores.get(qid)
                if cand_scores is None:
                    n_missing_qid += 1
                    continue
                score_control = t["control_score"]
                score_attack = t["attack_score"]
                rank_control = compute_rank(cand_scores, docid, score_control)
                rank_attack = compute_rank(cand_scores, docid, score_attack)
                delta_rank = rank_control - rank_attack
                writer.writerow({
                    "attack_name": spec.attack_name, "qid": qid, "docid": docid,
                    "score_control": score_control, "score_attack": score_attack,
                    "delta_score": score_attack - score_control,
                    "rank_control": rank_control, "rank_attack": rank_attack,
                    "delta_rank": delta_rank, "success": success_fn(delta_rank),
                })
                n_rows_total += 1
        print(f"  [{spec.attack_name}] {len(targets)} targets -> {out_path}")

    if n_missing_qid:
        print(f"[02_compute_score_rank] WARNING: {n_missing_qid} targets skipped "
              f"(qid not in the candidate pool).")
    print(f"[02_compute_score_rank] Done. {n_rows_total} total rows across {len(specs)} attacks.")
    print(f"\n  Next: python scripts/03_aggregate.py --config {args.config}")


if __name__ == "__main__":
    main()
