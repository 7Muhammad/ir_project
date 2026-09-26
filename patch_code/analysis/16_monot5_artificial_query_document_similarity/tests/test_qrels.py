"""Tests 17-19: qrel groups, within-query matching, seeded balancing."""

from __future__ import annotations

import pytest

from exp16lib.qrels import balance_query, build_balanced_manifest, relevance_group


def _qrels():
    q = []
    q += [("q1", f"r{i}", 2 if i % 2 else 3) for i in range(3)] + [("q1", f"n{i}", 0) for i in range(10)]
    q += [("q1", "g1", 1)]
    q += [("q2", f"r{i}", 3) for i in range(6)] + [("q2", f"n{i}", 0) for i in range(2)] + [("q2", "g1b", 1)]
    q += [("q3", "only_rel", 2), ("q3", "one", 1)]             # no grade-0 doc -> dropped
    return q


def test_17_group_definitions():
    assert relevance_group(3) == "relevant" and relevance_group(2) == "relevant"
    assert relevance_group(0) == "nonrelevant"
    assert relevance_group(1) is None
    with pytest.raises(ValueError):
        relevance_group(4)
    recs, summ = build_balanced_manifest(_qrels(), 42)
    assert all(r["qrel_grade"] in (2, 3) for r in recs if r["relevance_group"] == "relevant")
    assert all(r["qrel_grade"] == 0 for r in recs if r["relevance_group"] == "nonrelevant")
    assert not any(r["qrel_grade"] == 1 for r in recs)
    assert {s["qid"]: s["n_grade1_excluded"] for s in summ} == {"q1": 1, "q2": 1, "q3": 1}


def test_18_within_query_matching():
    recs, summ = build_balanced_manifest(_qrels(), 42)
    judged = {(q, d) for q, d, _ in _qrels()}
    assert all((r["qid"], r["docid"]) in judged for r in recs)   # never mixes docs across queries
    by_q = {}
    for r in recs:
        by_q.setdefault(r["qid"], {"relevant": 0, "nonrelevant": 0})[r["relevance_group"]] += 1
    assert set(by_q) == {"q1", "q2"}                              # q3 has no qrel-0 doc
    assert not [s for s in summ if s["qid"] == "q3"][0]["retained"]


def test_19_balancing_and_seed():
    recs, summ = build_balanced_manifest(_qrels(), 42)
    s = {x["qid"]: x for x in summ}
    assert s["q1"]["n_q"] == 3 and s["q2"]["n_q"] == 2
    q1 = [r for r in recs if r["qid"] == "q1"]
    assert sorted(r["docid"] for r in q1 if r["relevance_group"] == "relevant") == ["r0", "r1", "r2"]  # all rel kept
    assert sum(r["relevance_group"] == "nonrelevant" for r in q1) == 3
    q2 = [r for r in recs if r["qid"] == "q2"]
    assert sorted(r["docid"] for r in q2 if r["relevance_group"] == "nonrelevant") == ["n0", "n1"]     # all non kept
    assert sum(r["relevance_group"] == "relevant" for r in q2) == 2
    again, _ = build_balanced_manifest(list(reversed(_qrels())), 42)
    assert sorted(map(str, recs)) == sorted(map(str, again))       # seed 42 reproduces, order-independent
    r1, n1 = balance_query("q1", {f"r{i}": 2 for i in range(3)}, {f"n{i}": 0 for i in range(10)}, 42)
    r2, n2 = balance_query("q1", {f"r{i}": 2 for i in range(3)}, {f"n{i}": 0 for i in range(10)}, 7)
    assert r1 == r2 and len(n1) == len(n2) == 3
    other = [balance_query("q1", {}, {}, 42)]
    assert other == [([], [])]


def test_ineligible_docs_removed_before_balancing():
    elig = {(q, d) for q, d, _ in _qrels()} - {("q1", "r0")}
    recs, summ = build_balanced_manifest(_qrels(), 42, elig)
    s = {x["qid"]: x for x in summ}
    assert s["q1"]["n_q"] == 2 and s["q1"]["n_relevant_ineligible"] == 1
    assert ("q1", "r0") not in {(r["qid"], r["docid"]) for r in recs}
