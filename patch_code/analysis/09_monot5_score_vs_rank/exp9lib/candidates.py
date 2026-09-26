"""
exp9lib/candidates.py
========================
Build the fixed BM25 top-100 candidate set (DecoderLens's
`load_attack_tsv`/`build_candidate_sets`, see exp9lib/decoderlens_import.py)
and score every candidate's CLEAN input ONCE, using Experiment 1's own
scoring function (Type A: plain query+passage, no attack).

One-time cost: all 105 curated attacks share an identical 42-query set and
an identical top-100 BM25 candidate list per query (verified empirically —
the underlying BM25 run is replicated byte-for-byte into every attack TSV).
So this only needs to run once, against any single attack's TSV, and the
result is reused for every attack's rank computation.
"""

from __future__ import annotations

import pathlib
from collections import defaultdict
from typing import Callable, Dict, List, Tuple

from src.model_utils import build_monot5_input
from src.scoring import score_batch


def build_global_candidate_scores(
    model,
    tokenizer,
    true_id: int,
    false_id: int,
    max_length: int,
    device,
    sample_attack_tsv_path: pathlib.Path,
    candidate_top_k: int,
    load_attack_tsv: Callable,
    build_candidate_sets: Callable,
    batch_size: int = 8,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, list]]:
    """
    Returns (candidate_scores, candidate_sets):
      candidate_scores : {qid: {docid: clean_score}}
      candidate_sets    : {qid: [Candidate, ...]} (rank-sorted, len <= candidate_top_k)
        passed through for callers that need candidate metadata (e.g. counts).
    """
    records = load_attack_tsv(sample_attack_tsv_path)
    candidate_sets = build_candidate_sets(records, candidate_top_k)

    flat = [c for cands in candidate_sets.values() for c in cands]
    texts = [build_monot5_input(c.query, c.passage) for c in flat]
    print(f"[candidates] Scoring {len(flat)} candidates "
          f"({len(candidate_sets)} queries x up to {candidate_top_k}) ...")
    scores = score_batch(
        model, tokenizer, texts, true_id, false_id, max_length, device, batch_size
    )

    candidate_scores: Dict[str, Dict[str, float]] = defaultdict(dict)
    for cand, score in zip(flat, scores):
        candidate_scores[cand.qid][cand.docid] = score

    return dict(candidate_scores), candidate_sets
