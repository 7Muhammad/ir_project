#!/usr/bin/env python3
"""
scripts/03_aggregate.py
=========================
Reads (never modifies) the per-example raw CSVs written by
scripts/01_run_causal.py and scripts/02_run_attention.py, and writes the
aggregate tables listed in the experiment prompt's "OUTPUT — AGGREGATE
TABLES" section to outputs/aggregates/.

All aggregate tables are derived from the raw per-example outputs -- no
aggregate is computed on the fly inside the run scripts.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
EXP_DIR = SCRIPT_DIR.parent
EXP1_DIR = EXP_DIR.parent / "01_monot5_layer_patching"
EXP6_DIR = EXP_DIR.parent / "06_monot5_query_doc_patching"
sys.path.insert(0, str(EXP1_DIR))
sys.path.insert(0, str(EXP6_DIR))
sys.path.insert(0, str(EXP_DIR))

import pandas as pd

from exp12lib.aggregation import (
    example_balanced_aggregate,
    example_balanced_attention_aggregate,
    intervention_weighted_aggregate,
)
from exp12lib.run_utils import load_config, resolve_cfg_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Experiment 12 -- aggregate causal + attention outputs.")
    p.add_argument("--config", default=str(EXP_DIR / "configs" / "default.yaml"))
    return p.parse_args()


def load_all_results(base_dir: pathlib.Path, part: str) -> pd.DataFrame:
    attack_dirs = sorted((base_dir / part / "attacks").glob("*/results.csv"))
    if not attack_dirs:
        return pd.DataFrame()
    frames = [pd.read_csv(p) for p in attack_dirs]
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    args = parse_args()
    cfg = load_config(pathlib.Path(args.config))
    outputs_base = resolve_cfg_path(cfg, cfg["outputs"]["base_dir"])
    agg_dir = outputs_base / "aggregates"
    agg_dir.mkdir(parents=True, exist_ok=True)

    causal_df = load_all_results(outputs_base, "causal")
    attention_df = load_all_results(outputs_base, "attention")

    if causal_df.empty:
        print("[03_aggregate] No causal results found -- run scripts/01_run_causal.py first.")
    else:
        words = causal_df[causal_df["intervention_type"] == "query_word"].copy()
        structural = causal_df[causal_df["intervention_type"] == "structural_group"].copy()

        # 1. causal effects by attack x layer x query-word group
        t1 = example_balanced_aggregate(words, ["attack_name", "layer", "word_group"])
        t1.to_csv(agg_dir / "causal_by_attack_layer_wordgroup.csv", index=False)
        intervention_weighted_aggregate(words, ["attack_name", "layer", "word_group"]).to_csv(
            agg_dir / "causal_by_attack_layer_wordgroup_weighted.csv", index=False)

        # 2. causal effects by layer x query-word group, across all attacks (global)
        t2 = example_balanced_aggregate(words, ["layer", "word_group"])
        t2.to_csv(agg_dir / "causal_by_layer_wordgroup_global.csv", index=False)
        intervention_weighted_aggregate(words, ["layer", "word_group"]).to_csv(
            agg_dir / "causal_by_layer_wordgroup_global_weighted.csv", index=False)

        # 3. causal effects by attack x layer x structural group (+ global)
        if not structural.empty:
            t3 = example_balanced_aggregate(structural, ["attack_name", "layer", "intervention_name"])
            t3.to_csv(agg_dir / "causal_by_attack_layer_structural.csv", index=False)
            t3g = example_balanced_aggregate(structural, ["layer", "intervention_name"])
            t3g.to_csv(agg_dir / "causal_by_layer_structural_global.csv", index=False)

        # 7. ranked individual-query-word causal diagnostics (raw, sorted)
        rank_cols = ["attack_name", "qid", "docid", "layer", "query_word_index", "query_word_text",
                     "content_or_stopword", "matched_or_unmatched", "word_group",
                     "score_control", "score_attack", "delta",
                     "forward_effect", "reverse_effect", "combined_effect"]
        ranked = words[rank_cols].sort_values("combined_effect", ascending=False).reset_index(drop=True)
        ranked.insert(0, "rank_by_combined_effect_desc", ranked.index + 1)
        ranked.to_csv(agg_dir / "causal_query_word_ranked.csv", index=False)

        print(f"[03_aggregate] Causal: {len(causal_df)} raw rows "
              f"({len(words)} query_word, {len(structural)} structural_group) -> {agg_dir}")

    if attention_df.empty:
        print("[03_aggregate] No attention results found -- run scripts/02_run_attention.py first.")
    else:
        layer_avg = attention_df[attention_df["granularity"] == "layer_avg"].copy()
        important = attention_df[attention_df["granularity"] == "important_head"].copy()

        # 4. attention by attack x layer x query-word group (layer-averaged, all heads)
        a4 = example_balanced_attention_aggregate(layer_avg, ["attack_name", "layer", "word_group"])
        a4.to_csv(agg_dir / "attention_by_attack_layer_wordgroup.csv", index=False)

        # 5. attention across all attacks x layer x query-word group (global)
        a5 = example_balanced_attention_aggregate(layer_avg, ["layer", "word_group"])
        a5.to_csv(agg_dir / "attention_by_layer_wordgroup_global.csv", index=False)

        # 6. attention restricted to the previously identified important encoder heads
        if not important.empty:
            a6 = example_balanced_attention_aggregate(important, ["layer", "head", "word_group"])
            a6.to_csv(agg_dir / "attention_important_heads_by_layer_head_wordgroup.csv", index=False)
            a6_by_attack = example_balanced_attention_aggregate(
                important, ["attack_name", "layer", "head", "word_group"])
            a6_by_attack.to_csv(agg_dir / "attention_important_heads_by_attack_layer_head_wordgroup.csv", index=False)

        print(f"[03_aggregate] Attention: {len(attention_df)} raw rows "
              f"({len(layer_avg)} layer_avg, {len(important)} important_head) -> {agg_dir}")

    print(f"\n  Next: python scripts/04_make_plots.py --config {args.config}")


if __name__ == "__main__":
    main()
