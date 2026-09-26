#!/usr/bin/env python3
"""
scripts/01_build_candidate_scores.py
=======================================
One-time step: build the fixed BM25 top-100 candidate set (per query) and
score every candidate's CLEAN input, using Experiment 1's own scoring
function. All 105 curated attacks share an identical 42-query set and an
identical top-100 candidate list per query (verified — see DECISIONS.md),
so this runs against a single attack's TSV and the result is reused by
every attack's rank computation in script 02.

Output: outputs/candidates/candidate_scores.json
  {qid: {docid: clean_score}}
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
DECODERLENS_DIR = EXP_DIR.parent / "monot5_decoderlens_rank"
sys.path.insert(0, str(EXP1_DIR))   # src.*  (Experiment 1) — NOT DecoderLens's src
sys.path.insert(0, str(EXP3_DIR))   # headlib.*
sys.path.insert(0, str(EXP_DIR))    # exp9lib.*

from src.attack_registry import discover_attacks  # noqa: E402
from src.model_utils import get_true_false_token_ids, load_monot5, resolve_device  # noqa: E402

from exp9lib.candidates import build_global_candidate_scores  # noqa: E402
from exp9lib.decoderlens_import import load_decoderlens_ranking  # noqa: E402
from exp9lib.run_utils import load_config, resolve_cfg_path  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build the global candidate-score pool.")
    p.add_argument("--config", default=str(EXP_DIR / "configs" / "default.yaml"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(pathlib.Path(args.config))
    outputs_base = resolve_cfg_path(cfg, cfg["outputs"]["base_dir"])
    candidates_dir = outputs_base / "candidates"
    candidates_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(cfg["model"]["device"])
    model, tokenizer = load_monot5(cfg["model"]["checkpoint"], device)
    true_id, false_id = get_true_false_token_ids(tokenizer)

    specs = discover_attacks(cfg["attacks"])
    if not specs:
        sys.exit("No attacks discovered from config — check attacks.inherit_from.")
    sample_spec = specs[0]
    print(f"[01_build_candidate_scores] Using {sample_spec.attack_name}'s TSV to build "
          f"the candidate set ({len(specs)} attacks share it identically).")

    _Candidate, load_attack_tsv, build_candidate_sets, _compute_rank = (
        load_decoderlens_ranking(DECODERLENS_DIR)
    )

    candidate_scores, candidate_sets = build_global_candidate_scores(
        model=model, tokenizer=tokenizer, true_id=true_id, false_id=false_id,
        max_length=cfg["model"]["max_length"], device=device,
        sample_attack_tsv_path=sample_spec.path,
        candidate_top_k=cfg["ranking"]["candidate_top_k"],
        load_attack_tsv=load_attack_tsv, build_candidate_sets=build_candidate_sets,
        batch_size=cfg["runtime"]["batch_size_scoring"],
    )

    out_path = candidates_dir / "candidate_scores.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(candidate_scores, fh)

    n_queries = len(candidate_scores)
    n_candidates = sum(len(v) for v in candidate_scores.values())
    print(f"[01_build_candidate_scores] {n_queries} queries, {n_candidates} candidates "
          f"scored -> {out_path}")


if __name__ == "__main__":
    main()
