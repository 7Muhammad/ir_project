"""
src/data_loading.py
===================
Load a pre-built attacked-document TSV and build the per-query candidate sets
used for rank computation.

Attacked TSV format (tab-separated, with header)
------------------------------------------------
    qid  query  docno  score  rank  text  text_0

Fields:
    text   — attacked passage (injection tokens added)
    text_0 — original clean passage

Candidate set
-------------
For each query we keep the top-k candidates by BM25 rank (the ``rank`` column
already in the TSV).  We never recompute BM25.  Every candidate carries its
ORIGINAL clean passage (text_0).  The candidate set is the fixed pool against
which a target passage's rank is computed; only the target passage is swapped
for one variant (original / padded_control / attack), all other candidates
stay clean.
"""

from __future__ import annotations

import csv
import pathlib
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class Candidate:
    qid: str
    docid: str
    rank: int          # BM25 rank (1 = best)
    query: str
    passage: str       # clean passage (text_0)
    attacked_passage: str  # attacked passage (text)


def load_attack_tsv(path: pathlib.Path) -> List[Candidate]:
    """Load every row of the attacked TSV into a list of Candidate records."""
    # The MS MARCO passages can be long; lift the CSV field-size limit.
    try:
        csv.field_size_limit(10 * 1024 * 1024)
    except OverflowError:  # pragma: no cover - platform dependent
        csv.field_size_limit(2 ** 31 - 1)

    records: List[Candidate] = []
    with open(path, "r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            try:
                rank = int(row.get("rank", -1))
            except (TypeError, ValueError):
                rank = -1
            records.append(
                Candidate(
                    qid=str(row.get("qid", "")),
                    docid=str(row.get("docno") or row.get("docid", "")),
                    rank=rank,
                    query=str(row.get("query", "")),
                    passage=str(row.get("text_0", "")),
                    attacked_passage=str(row.get("text", "")),
                )
            )
    return records


def build_candidate_sets(
    records: List[Candidate], candidate_top_k: int
) -> Dict[str, List[Candidate]]:
    """
    Group candidates by qid and keep the top-k by BM25 rank.

    Returns a dict {qid: [Candidate, ...]} sorted by ascending rank.
    """
    by_qid: Dict[str, List[Candidate]] = defaultdict(list)
    for r in records:
        by_qid[r.qid].append(r)

    candidate_sets: Dict[str, List[Candidate]] = {}
    for qid, cands in by_qid.items():
        cands_sorted = sorted(
            cands, key=lambda c: c.rank if c.rank >= 0 else 10 ** 9
        )
        candidate_sets[qid] = cands_sorted[:candidate_top_k]
    return candidate_sets


def iter_target_candidates(
    candidate_sets: Dict[str, List[Candidate]]
) -> List[Candidate]:
    """
    Flatten the candidate sets into a single ordered list of potential targets.

    Targets are drawn across queries in a round-robin fashion so that, when the
    selection cap is small, the kept targets span many queries rather than a
    single one.
    """
    qids = sorted(candidate_sets.keys())
    # Round-robin over per-query candidate lists (already rank-sorted).
    flattened: List[Candidate] = []
    max_len = max((len(candidate_sets[q]) for q in qids), default=0)
    for i in range(max_len):
        for q in qids:
            cands = candidate_sets[q]
            if i < len(cands):
                flattened.append(cands[i])
    return flattened
