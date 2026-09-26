#!/usr/bin/env python3
"""
scripts/05_diagnostics.py
===========================
Individual query-word diagnostic table (see experiment prompt's "INDIVIDUAL
QUERY-WORD DIAGNOSTIC" section): lets us inspect, for one example, each
query word's combined causal effect at every encoder layer --

    how      combined = ...
    long     combined = ...
    do       combined = ...
    fleas    combined = ...
    live     combined = ...

This is for QUALITATIVE inspection only -- never pooled into a global claim
about "position N" across unrelated queries.

Outputs
-------
outputs/diagnostics/individual_query_word_table.csv
    Full long-format table: one row per (attack, qid, docid, layer, word).
outputs/diagnostics/examples/<attack>__<qid>__<docid>.csv
    A handful of example-level pivots (rows=layer, cols=query word, by
    query-position order) for the highest attack_delta examples, for quick
    eyeballing -- reads exactly like the "how | long | do | fleas | live"
    illustration in the prompt.
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
EXP_DIR = SCRIPT_DIR.parent
EXP1_DIR = EXP_DIR.parent / "01_monot5_layer_patching"
EXP6_DIR = EXP_DIR.parent / "06_monot5_query_doc_patching"
sys.path.insert(0, str(EXP1_DIR))
sys.path.insert(0, str(EXP6_DIR))
sys.path.insert(0, str(EXP_DIR))

import pandas as pd

from exp12lib.run_utils import load_config, resolve_cfg_path

_SAFE = re.compile(r"[^A-Za-z0-9_.-]+")


def _safe(s: str) -> str:
    return _SAFE.sub("_", str(s))[:80]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Experiment 12 -- individual query-word diagnostic table.")
    p.add_argument("--config", default=str(EXP_DIR / "configs" / "default.yaml"))
    p.add_argument("--n-example-pivots", type=int, default=15,
                    help="Number of highest-delta examples to also emit as readable per-example pivots.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(pathlib.Path(args.config))
    outputs_base = resolve_cfg_path(cfg, cfg["outputs"]["base_dir"])
    causal_files = sorted((outputs_base / "causal" / "attacks").glob("*/results.csv"))
    if not causal_files:
        sys.exit("[05_diagnostics] No causal results found -- run scripts/01_run_causal.py first.")

    df = pd.concat([pd.read_csv(p) for p in causal_files], ignore_index=True)
    words = df[df["intervention_type"] == "query_word"].copy()

    diag_dir = outputs_base / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)

    long_cols = ["attack_name", "qid", "docid", "layer", "query_word_index", "query_word_text",
                 "content_or_stopword", "matched_or_unmatched", "word_group",
                 "delta", "forward_effect", "reverse_effect", "combined_effect"]
    long_table = words[long_cols].sort_values(
        ["attack_name", "qid", "docid", "layer", "query_word_index"]
    ).reset_index(drop=True)
    long_table.to_csv(diag_dir / "individual_query_word_table.csv", index=False)
    print(f"[05_diagnostics] wrote {diag_dir / 'individual_query_word_table.csv'} ({len(long_table)} rows)")

    examples_dir = diag_dir / "examples"
    examples_dir.mkdir(parents=True, exist_ok=True)
    top_examples = (
        words[["attack_name", "qid", "docid", "delta"]].drop_duplicates()
        .sort_values("delta", ascending=False).head(args.n_example_pivots)
    )
    for _, ex in top_examples.iterrows():
        sub = words[(words.attack_name == ex.attack_name) & (words.qid == ex.qid) & (words.docid == ex.docid)]
        word_order = (sub[["query_word_index", "query_word_text"]].drop_duplicates()
                      .sort_values("query_word_index"))
        col_labels = {row.query_word_index: f"{row.query_word_text} (w{int(row.query_word_index)})"
                      for row in word_order.itertuples()}
        pivot = sub.pivot(index="layer", columns="query_word_index", values="combined_effect").sort_index()
        pivot = pivot.rename(columns=col_labels)
        fname = f"{_safe(ex.attack_name)}__{_safe(ex.qid)}__{_safe(ex.docid)}.csv"
        pivot.to_csv(examples_dir / fname)

    print(f"[05_diagnostics] wrote {len(top_examples)} example pivot(s) to {examples_dir}")


if __name__ == "__main__":
    main()
