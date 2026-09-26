"""
exp12lib/aggregation.py
=========================
Example-balanced aggregation, shared by scripts/03_aggregate.py and
scripts/05_diagnostics.py.

Example-balanced procedure (main result, per the experiment prompt)
------------------------------------------------------------------------
A query (e.g. 8 words) should not outweigh a query (e.g. 2 words) just
because it has more words in a given group. So, per `example_cols` (an
example = one (attack_name, qid, docid)) and `group_cols` (e.g.
[layer, word_group] or [attack_name, layer, word_group]):

  1. average every raw row belonging to the SAME (example, group) --
     e.g. all "content_matched" words in one example, at one layer.
  2. average those per-example values across examples.

The (unweighted-by-word-count) intervention-weighted variant skips step 1
and averages raw rows directly -- kept as a secondary diagnostic per the
prompt ("Also save intervention-weighted aggregates as secondary
diagnostics if useful").
"""

from __future__ import annotations

from typing import List

import pandas as pd

EXAMPLE_COLS = ["attack_name", "qid", "docid"]


def example_balanced_aggregate(
    df: pd.DataFrame, group_cols: List[str], value_col: str = "combined_effect",
    extra_value_cols: List[str] = ("forward_effect", "reverse_effect"),
) -> pd.DataFrame:
    """
    Two-step example-balanced aggregate over `group_cols` (which may or may
    not include "attack_name" -- omit it for the global, cross-attack table).

    Returns columns: group_cols + [mean_forward, mean_reverse, mean_combined,
    median_combined, std_combined, n_examples, n_word_interventions].
    """
    all_value_cols = [value_col] + list(extra_value_cols)
    per_example_cols = list(dict.fromkeys(EXAMPLE_COLS + group_cols))  # dedup, preserve order

    n_interventions = df.groupby(group_cols, dropna=False).size().rename("n_word_interventions")

    step1 = df.groupby(per_example_cols, dropna=False)[all_value_cols].mean().reset_index()

    step2 = step1.groupby(group_cols, dropna=False).agg(
        mean_forward=("forward_effect", "mean"),
        mean_reverse=("reverse_effect", "mean"),
        mean_combined=(value_col, "mean"),
        median_combined=(value_col, "median"),
        std_combined=(value_col, "std"),
        n_examples=(value_col, "size"),
    ).reset_index()

    return step2.merge(n_interventions, on=group_cols, how="left")


def intervention_weighted_aggregate(
    df: pd.DataFrame, group_cols: List[str], value_col: str = "combined_effect",
    extra_value_cols: List[str] = ("forward_effect", "reverse_effect"),
) -> pd.DataFrame:
    """Secondary diagnostic: raw-row average, no per-example word-count balancing."""
    agg = df.groupby(group_cols, dropna=False).agg(
        mean_forward=("forward_effect", "mean"),
        mean_reverse=("reverse_effect", "mean"),
        mean_combined=(value_col, "mean"),
        median_combined=(value_col, "median"),
        std_combined=(value_col, "std"),
        n_word_interventions=(value_col, "size"),
    ).reset_index()
    return agg


def example_balanced_attention_aggregate(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    """
    Same two-step example-balancing, applied to attention delta columns
    instead of causal effect columns.
    """
    value_cols = ["attack_to_query_delta", "query_to_attack_delta",
                  "attack_to_query_control", "attack_to_query_attack",
                  "query_to_attack_control", "query_to_attack_attack"]
    per_example_cols = list(dict.fromkeys(EXAMPLE_COLS + group_cols))

    n_word_rows = df.groupby(group_cols, dropna=False).size().rename("n_word_observations")

    step1 = df.groupby(per_example_cols, dropna=False)[value_cols].mean().reset_index()
    step2 = step1.groupby(group_cols, dropna=False).agg(
        mean_attack_to_query_delta=("attack_to_query_delta", "mean"),
        median_attack_to_query_delta=("attack_to_query_delta", "median"),
        std_attack_to_query_delta=("attack_to_query_delta", "std"),
        mean_query_to_attack_delta=("query_to_attack_delta", "mean"),
        median_query_to_attack_delta=("query_to_attack_delta", "median"),
        std_query_to_attack_delta=("query_to_attack_delta", "std"),
        mean_attack_to_query_control=("attack_to_query_control", "mean"),
        mean_attack_to_query_attack=("attack_to_query_attack", "mean"),
        mean_query_to_attack_control=("query_to_attack_control", "mean"),
        mean_query_to_attack_attack=("query_to_attack_attack", "mean"),
        n_examples=("attack_to_query_delta", "size"),
    ).reset_index()
    return step2.merge(n_word_rows, on=group_cols, how="left")
