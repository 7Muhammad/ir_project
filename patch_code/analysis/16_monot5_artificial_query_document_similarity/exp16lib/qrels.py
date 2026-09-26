"""
exp16lib/qrels.py
==================
Human-relevance groups and balanced WITHIN-QUERY sampling (DECISIONS 10-18).

  grade 2 or 3 -> "relevant"      grade 0 -> "nonrelevant"      grade 1 -> excluded

For each query q with both groups non-empty:
    n_q = min(|R_q|, |N_q|)
    |R_q| <= |N_q| : all of R_q + a seeded sample of n_q from N_q
    |R_q| >  |N_q| : a seeded sample of n_q from R_q + all of N_q
Sampling uses random.Random(f"{seed}:{qid}") over the docids sorted as
strings, so each query's sample is reproducible and independent of the
iteration order of other queries. monoT5 scores play no role here.
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

RELEVANT, NONRELEVANT = "relevant", "nonrelevant"


def relevance_group(grade: int) -> Optional[str]:
    grade = int(grade)
    if grade in (2, 3):
        return RELEVANT
    if grade == 0:
        return NONRELEVANT
    if grade == 1:
        return None
    raise ValueError(f"unexpected qrel grade {grade}")


def group_by_query(qrels: Iterable[Tuple[str, str, int]]) -> Dict[str, Dict[str, Dict[str, int]]]:
    """(qid, docid, grade) -> {qid: {"relevant": {docid: grade}, "nonrelevant": {...}, "excluded": {...}}}"""
    out: Dict[str, Dict[str, Dict[str, int]]] = defaultdict(lambda: {RELEVANT: {}, NONRELEVANT: {}, "excluded": {}})
    for qid, docid, grade in qrels:
        g = relevance_group(grade)
        out[str(qid)][g or "excluded"][str(docid)] = int(grade)
    return dict(out)


def balance_query(qid: str, relevant: Dict[str, int], nonrelevant: Dict[str, int], seed: int
                  ) -> Tuple[List[str], List[str]]:
    """Returns (selected relevant docids, selected nonrelevant docids), both sorted, equal length."""
    R, N = sorted(relevant), sorted(nonrelevant)
    n = min(len(R), len(N))
    if n == 0:
        return [], []
    rng = random.Random(f"{seed}:{qid}")
    if len(R) <= len(N):
        return R, sorted(rng.sample(N, n))
    return sorted(rng.sample(R, n)), N


def build_balanced_manifest(qrels: Iterable[Tuple[str, str, int]], seed: int,
                            eligible_docs: Optional[set] = None) -> Tuple[List[Dict], List[Dict]]:
    """
    Returns (records, per-query summary). `eligible_docs`, if given, is the set
    of (qid, docid) pairs usable at all (e.g. text available, prompt fits);
    ineligible judged docs are removed BEFORE balancing and counted.
    """
    groups = group_by_query(qrels)
    records, summary = [], []
    for qid in sorted(groups, key=lambda x: (len(x), x)):
        g = groups[qid]
        rel = {d: v for d, v in g[RELEVANT].items() if eligible_docs is None or (qid, d) in eligible_docs}
        non = {d: v for d, v in g[NONRELEVANT].items() if eligible_docs is None or (qid, d) in eligible_docs}
        sel_r, sel_n = balance_query(qid, rel, non, seed)
        summary.append({
            "qid": qid, "n_judged": len(g[RELEVANT]) + len(g[NONRELEVANT]) + len(g["excluded"]),
            "n_relevant_available": len(rel), "n_nonrelevant_available": len(non),
            "n_grade1_excluded": len(g["excluded"]),
            "n_relevant_ineligible": len(g[RELEVANT]) - len(rel),
            "n_nonrelevant_ineligible": len(g[NONRELEVANT]) - len(non),
            "n_q": len(sel_r), "retained": bool(sel_r),
            "n_grade2_selected": sum(1 for d in sel_r if rel[d] == 2),
            "n_grade3_selected": sum(1 for d in sel_r if rel[d] == 3),
        })
        for d in sel_r:
            records.append({"qid": qid, "docid": d, "qrel_grade": rel[d], "relevance_group": RELEVANT})
        for d in sel_n:
            records.append({"qid": qid, "docid": d, "qrel_grade": non[d], "relevance_group": NONRELEVANT})
    return records, summary
