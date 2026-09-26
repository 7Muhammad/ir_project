#!/usr/bin/env python3
"""
scripts/04_compute_layerwise_ranks.py
=====================================
Turn layerwise scores into layerwise RANKS.

For each target example, variant, and encoder layer we rank the target against
the WHOLE candidate set: every other candidate keeps its clean score and only
the target's score is replaced by the variant score.  Rank 1 is best.

Outputs
-------
  ranks/layerwise_target_ranks.csv
  ranks/layerwise_rank_summary.csv

Definitions
-----------
  rank_gain_vs_control  = control_rank - variant_rank
      ( > 0  -> variant improved rank ;  < 0  -> variant hurt rank )
  score_delta_vs_control = variant_score - control_score
  success_rate_vs_control       = fraction with variant_rank  < control_rank
  score_success_rate_vs_control = fraction with variant_score > control_score

Usage
-----
  python scripts/04_compute_layerwise_ranks.py --config configs/default.yaml \
      --attack-name relevant_start_5
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from typing import Dict, List

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.attack_registry import AttackSpec, discover_attacks
from src.ranking import compute_rank
from src.utils import attack_output_dirs, layer_name, load_config


def _candidate_lookup(cand_df: pd.DataFrame) -> Dict[str, Dict[int, Dict[str, float]]]:
    """Build {qid: {layer_index: {docid: clean_score}}} from the candidate parquet."""
    lookup: Dict[str, Dict[int, Dict[str, float]]] = {}
    for (qid, layer), grp in cand_df.groupby(["qid", "layer_index"]):
        lookup.setdefault(str(qid), {})[int(layer)] = dict(
            zip(grp["docid"].astype(str), grp["score"].astype(float))
        )
    return lookup


def rank_attack(spec: AttackSpec, cfg: dict, base_dir: pathlib.Path) -> Dict:
    """Compute ranks + summary for one attack from its scored outputs."""
    out_dirs = attack_output_dirs(base_dir, spec.attack_name)
    scores_csv = out_dirs["scores"] / "layerwise_target_scores.csv"
    cand_parquet = out_dirs["scores"] / "layerwise_candidate_scores.parquet"
    if not scores_csv.exists():
        raise FileNotFoundError(f"Missing {scores_csv}. Run Stage 03 first.")
    if not cand_parquet.exists():
        raise FileNotFoundError(
            f"Missing {cand_parquet}. Run Stage 03 with runtime.save_candidate_scores=true."
        )

    scores_df = pd.read_csv(scores_csv, dtype={"qid": str, "docid": str})
    if scores_df.empty:
        print(f"[04] {spec.attack_name}: empty scores, skipping.")
        return {"n_examples": 0}
    cand_df = pd.read_parquet(cand_parquet)
    cand_df["qid"] = cand_df["qid"].astype(str)
    cand_df["docid"] = cand_df["docid"].astype(str)
    lookup = _candidate_lookup(cand_df)

    candidate_top_k = int(cfg["ranking"]["candidate_top_k"])

    # --- Per-row ranks -------------------------------------------------------
    rank_rows: List[Dict] = []
    for r in scores_df.itertuples(index=False):
        qid, docid = str(r.qid), str(r.docid)
        layer = int(r.layer_index)
        cand_scores = lookup.get(qid, {}).get(layer, {})
        rank = compute_rank(cand_scores, docid, float(r.score))
        rank_rows.append({
            "attack_name": spec.attack_name,
            "token": spec.token,
            "position": spec.position,
            "repetitions": spec.repetitions,
            "qid": qid,
            "docid": docid,
            "variant": r.variant,
            "layer_index": layer,
            "layer_name": layer_name(layer),
            "score": float(r.score),
            "rank": rank,
            "candidate_top_k": candidate_top_k,
            "n_candidates_scored": len(cand_scores),
        })
    ranks_df = pd.DataFrame(rank_rows)
    ranks_csv = out_dirs["ranks"] / "layerwise_target_ranks.csv"
    ranks_df.to_csv(ranks_csv, index=False)
    print(f"[04] {spec.attack_name}: wrote {len(ranks_df)} rank rows -> {ranks_csv}")

    # --- Merge in padded_control reference for vs-control metrics -------------
    control = (
        ranks_df[ranks_df["variant"] == "padded_control"]
        [["qid", "docid", "layer_index", "rank", "score"]]
        .rename(columns={"rank": "control_rank", "score": "control_score"})
    )
    merged = ranks_df.merge(control, on=["qid", "docid", "layer_index"], how="left")
    merged["rank_gain_vs_control"] = merged["control_rank"] - merged["rank"]
    merged["score_delta_vs_control"] = merged["score"] - merged["control_score"]
    merged["rank_success"] = (merged["rank"] < merged["control_rank"]).astype(float)
    merged["score_success"] = (merged["score"] > merged["control_score"]).astype(float)

    # --- Aggregate summary ---------------------------------------------------
    summary_rows: List[Dict] = []
    for (variant, layer), grp in merged.groupby(["variant", "layer_index"]):
        summary_rows.append({
            "attack_name": spec.attack_name,
            "token": spec.token,
            "position": spec.position,
            "repetitions": spec.repetitions,
            "variant": variant,
            "layer_index": int(layer),
            "layer_name": layer_name(int(layer)),
            "n_examples": int(len(grp)),
            "mean_rank": float(grp["rank"].mean()),
            "median_rank": float(grp["rank"].median()),
            "mean_score": float(grp["score"].mean()),
            "median_score": float(grp["score"].median()),
            "success_rate_vs_control": float(grp["rank_success"].mean()),
            "score_success_rate_vs_control": float(grp["score_success"].mean()),
            "mean_rank_gain_vs_control": float(grp["rank_gain_vs_control"].mean()),
            "median_rank_gain_vs_control": float(grp["rank_gain_vs_control"].median()),
            "mean_score_delta_vs_control": float(grp["score_delta_vs_control"].mean()),
            "median_score_delta_vs_control": float(grp["score_delta_vs_control"].median()),
        })
    summary_df = pd.DataFrame(summary_rows).sort_values(["variant", "layer_index"])
    summary_csv = out_dirs["ranks"] / "layerwise_rank_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print(f"[04] {spec.attack_name}: wrote summary -> {summary_csv}")

    return {"n_examples": int(merged[merged["variant"] == "attack"]["qid"].nunique())}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute DecoderLens layerwise ranks.")
    p.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "default.yaml"))
    p.add_argument("--attack-name", default=None,
                   help="Single attack to rank (default: all discovered).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    base_dir = PROJECT_ROOT / cfg["outputs"]["base_dir"]
    attacks = discover_attacks(cfg["attacks"])
    if args.attack_name:
        attacks = [a for a in attacks if a.attack_name == args.attack_name]
        if not attacks:
            raise SystemExit(f"Attack '{args.attack_name}' not found.")
    for spec in attacks:
        rank_attack(spec, cfg, base_dir)


if __name__ == "__main__":
    main()
