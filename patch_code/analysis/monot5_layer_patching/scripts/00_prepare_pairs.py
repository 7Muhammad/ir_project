#!/usr/bin/env python3
"""
scripts/00_prepare_pairs.py
============================
Stage 00: Locate data files and produce outputs/pairs/pairs.jsonl.

Goal
----
Produce a normalised JSONL file where each record has:
    qid, docid, rank, bm25_score, query,
    passage          — original clean passage (text_0)
    attacked_passage — pre-built attacked passage (text) from the repo

The BM25 candidates and pre-built attacked documents come from the original
ECIR-24 adversarial-evaluation pipeline.  We never rerun BM25 or reconstruct
attacks; we load the exact files produced by the original paper's pipeline.

Priority
--------
1. If config.attack.attacked_pairs_path is set → load it directly (Mode A).
   This is the primary mode.  The attacked TSV (e.g. relevant_start_5_bm25_19.gz.tsv)
   contains both 'text_0' (clean passage) and 'text' (attacked passage).

2. If config.data.upstream_repo_path is set → look for the attacked TSV
   inside the repo's runs/injected/dl19/ directory.

3. If config.data.prepared_pairs_path is set → load a plain pairs file
   (no attacked_passage — later stages will error if they need it).

4. Otherwise → fall back to bm25_19.jsonl (clean only).
"""

from __future__ import annotations

import argparse
import pathlib
import sys

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

import yaml

from src.data_utils import (
    load_attacked_tsv,
    load_passage_collection,
    load_prepared_pairs,
    load_queries,
    parse_trec_run,
    reconstruct_pairs_from_run,
    save_pairs,
)
from src.find_data import (
    discover_data,
    resolve_single_candidate,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stage 00: prepare query-document pairs for the patching experiment."
    )
    p.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "default.yaml"),
        help="Path to the YAML config file.",
    )
    p.add_argument(
        "--attack-name",
        default=None,
        help="If set, write outputs to outputs/attacks/{attack-name}/ instead of outputs/.",
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
    print(f"[00_prepare_pairs] Loading config from: {cfg_path}")

    with open(cfg_path) as fh:
        cfg = yaml.safe_load(fh)

    data_cfg    = cfg["data"]
    attack_cfg  = cfg["attack"]
    runtime_cfg = cfg["runtime"]

    base_dir  = _resolve_base_dir(cfg, args.attack_name)
    pairs_out = base_dir / "pairs" / "pairs.jsonl"
    pairs_out.parent.mkdir(parents=True, exist_ok=True)

    max_pairs          = data_cfg["max_pairs"]
    max_docs_per_query = data_cfg["max_docs_per_query"]
    seed               = runtime_cfg["seed"]

    # -------------------------------------------------------------------------
    # Mode A (primary): load pre-built attacked TSV from the repo
    # This gives us both 'passage' (clean) and 'attacked_passage' in one file.
    # -------------------------------------------------------------------------
    attacked_path_str = attack_cfg.get("attacked_pairs_path")
    if attacked_path_str:
        attacked_path = pathlib.Path(attacked_path_str).resolve()
        if not attacked_path.exists():
            sys.exit(
                f"ERROR: attacked_pairs_path '{attacked_path}' does not exist.\n"
                "Check the path in configs/default.yaml under attack.attacked_pairs_path.\n"
                "Expected location: ecir24-adversarial-evaluation/runs/injected/dl19/"
                "relevant_start_5_bm25_19.gz.tsv"
            )
        print(f"[00_prepare_pairs] Mode A: loading attacked TSV from {attacked_path}")
        pairs = load_attacked_tsv(attacked_path, max_pairs, max_docs_per_query, seed)
        save_pairs(pairs, pairs_out)
        print(f"[00_prepare_pairs] Done. Output: {pairs_out}")
        print(f"  Fields: {list(pairs[0].keys()) if pairs else '(empty)'}")
        return

    # -------------------------------------------------------------------------
    # Mode A via upstream_repo_path: find the attacked TSV automatically
    # -------------------------------------------------------------------------
    upstream = data_cfg.get("upstream_repo_path")
    if upstream:
        upstream_p = pathlib.Path(upstream).resolve()
        # Primary target: the exact attacked file used in this experiment
        candidate = upstream_p / "runs" / "injected" / "dl19" / "relevant_start_5_bm25_19.gz.tsv"
        if candidate.exists():
            print(f"[00_prepare_pairs] Mode A (upstream): loading attacked TSV {candidate}")
            pairs = load_attacked_tsv(candidate, max_pairs, max_docs_per_query, seed)
            save_pairs(pairs, pairs_out)
            print(f"[00_prepare_pairs] Done. Output: {pairs_out}")
            return
        # Fallback: clean pairs only from bm25_19.jsonl
        clean_candidate = upstream_p / "data" / "bm25_19.jsonl"
        if clean_candidate.exists():
            print(
                f"[00_prepare_pairs] WARNING: attacked TSV not found at {candidate}.\n"
                f"  Falling back to clean-only pairs from {clean_candidate}.\n"
                f"  NOTE: 'attacked_passage' will be missing — Stage 01 will fail.\n"
                f"  Set attack.attacked_pairs_path in the config to fix this."
            )
            pairs = load_prepared_pairs(clean_candidate, max_pairs, max_docs_per_query, seed)
            save_pairs(pairs, pairs_out)
            print(f"[00_prepare_pairs] Done (clean only). Output: {pairs_out}")
            return
        print(
            f"[00_prepare_pairs] WARNING: upstream_repo_path is set but neither "
            f"the attacked TSV nor bm25_19.jsonl was found. Trying other modes."
        )

    # -------------------------------------------------------------------------
    # Mode B: user-specified prepared pairs file (clean only)
    # -------------------------------------------------------------------------
    if data_cfg.get("prepared_pairs_path"):
        prepared_path = pathlib.Path(data_cfg["prepared_pairs_path"]).resolve()
        print(f"[00_prepare_pairs] Mode B: loading plain pairs from {prepared_path}")
        if not prepared_path.exists():
            sys.exit(
                f"ERROR: prepared_pairs_path '{prepared_path}' does not exist."
            )
        pairs = load_prepared_pairs(prepared_path, max_pairs, max_docs_per_query, seed)
        save_pairs(pairs, pairs_out)
        print(f"[00_prepare_pairs] Done (clean only — no attacked_passage). Output: {pairs_out}")
        return

    # -------------------------------------------------------------------------
    # Mode C: reconstruct from BM25 TREC run + queries + passages
    # -------------------------------------------------------------------------
    if data_cfg.get("bm25_run_path"):
        bm25_run_p = pathlib.Path(data_cfg["bm25_run_path"]).resolve()
        print(f"[00_prepare_pairs] Mode C: using TREC run {bm25_run_p}")

        # Queries
        if not data_cfg.get("queries_path"):
            sys.exit(
                "ERROR: bm25_run_path is set, but queries_path is not set in the config.\n"
                "Please provide the path to the DL19 queries file."
            )
        queries_p = pathlib.Path(data_cfg["queries_path"]).resolve()
        queries = load_queries(queries_p)

        # Passage collection
        if not data_cfg.get("passages_path"):
            sys.exit(
                "ERROR: bm25_run_path is set, but passages_path is not set in the config.\n"
                "A passage collection/docstore is required to look up passage text.\n"
                "Set 'passages_path' to the MS MARCO passage collection file (TSV or JSONL)."
            )
        passages_p = pathlib.Path(data_cfg["passages_path"]).resolve()

        run = parse_trec_run(bm25_run_p, max_docs_per_query)

        # Pre-compute needed docids to avoid loading the entire collection
        needed_docids = {doc["docid"] for docs in run.values() for doc in docs}
        passages = load_passage_collection(passages_p, needed_docids=needed_docids)

        if not passages:
            sys.exit(
                "ERROR: No passage texts were loaded from the passage collection.\n"
                "Check that passages_path points to a valid MS MARCO passage file."
            )

        pairs = reconstruct_pairs_from_run(run, queries, passages, max_pairs, seed)
        save_pairs(pairs, pairs_out)
        print(f"[00_prepare_pairs] Done. Output: {pairs_out}")
        return

    # -------------------------------------------------------------------------
    # Fallback: auto-discovery
    # -------------------------------------------------------------------------
    print(
        "[00_prepare_pairs] No prepared_pairs_path or bm25_run_path set.\n"
        "Running data discovery to find candidate files ...\n"
    )
    discovery_report = base_dir / "data_discovery" / "data_candidates.txt"
    candidates = discover_data(
        upstream_repo_path=upstream,
        search_roots=data_cfg.get("search_roots", []),
        report_path=discovery_report,
        script_dir=PROJECT_ROOT,
    )

    # Try to auto-resolve a prepared pairs file
    pairs_path = resolve_single_candidate(
        candidates["pairs"],
        category="prepared_pairs",
        config_override=data_cfg.get("prepared_pairs_path"),
    )

    if pairs_path is not None:
        print(f"[00_prepare_pairs] Auto-resolved pairs file: {pairs_path}")
        pairs = load_prepared_pairs(pairs_path, max_pairs, max_docs_per_query, seed)
        save_pairs(pairs, pairs_out)
        print(f"[00_prepare_pairs] Done. Output: {pairs_out}")
        return

    # Could not auto-resolve
    bm25_candidates = candidates.get("bm25_run", [])
    passage_candidates = candidates.get("passages", [])

    print("\n[00_prepare_pairs] Could not automatically determine the data source.")
    print("Discovery report saved to:", discovery_report)
    print(
        "\nNext steps:\n"
        "  Option 1 (easiest): Set 'upstream_repo_path' in configs/default.yaml\n"
        "    to the path of the cloned ECIR-24 repo.  The script will find\n"
        "    bm25_19.jsonl automatically.\n"
        "\n"
        "  Option 2: Set 'prepared_pairs_path' in configs/default.yaml\n"
        "    to a JSONL/CSV/TSV file with columns: qid, docid, query, passage.\n"
        "\n"
        "  Option 3: Set 'bm25_run_path' + 'queries_path' + 'passages_path'\n"
        "    in configs/default.yaml to reconstruct pairs from scratch.\n"
    )
    sys.exit(1)


if __name__ == "__main__":
    main()
