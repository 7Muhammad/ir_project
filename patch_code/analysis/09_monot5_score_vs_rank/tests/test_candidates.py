"""Tests for exp9lib/candidates.py — flatten/dedup logic, with score_batch
monkeypatched (a real tokenizer would be needed otherwise; that path is
covered by the smoke test instead)."""

from __future__ import annotations

from dataclasses import dataclass

import exp9lib.candidates as candidates_mod


@dataclass
class FakeCandidate:
    qid: str
    docid: str
    rank: int
    query: str
    passage: str
    attacked_passage: str


def fake_load_attack_tsv(path):
    return [
        FakeCandidate("q1", "d1", 1, "query one", "passage d1", "atk d1"),
        FakeCandidate("q1", "d2", 2, "query one", "passage d2", "atk d2"),
        FakeCandidate("q2", "d3", 1, "query two", "passage d3", "atk d3"),
    ]


def fake_build_candidate_sets(records, candidate_top_k):
    by_qid = {}
    for r in records:
        by_qid.setdefault(r.qid, []).append(r)
    return {qid: sorted(cands, key=lambda c: c.rank)[:candidate_top_k]
            for qid, cands in by_qid.items()}


def test_build_global_candidate_scores_flattens_and_groups(monkeypatch):
    calls = {}

    def fake_score_batch(model, tokenizer, texts, true_id, false_id, max_length, device, batch_size):
        calls["texts"] = texts
        return [float(i) for i in range(len(texts))]  # deterministic, index-based

    monkeypatch.setattr(candidates_mod, "score_batch", fake_score_batch)

    candidate_scores, candidate_sets = candidates_mod.build_global_candidate_scores(
        model=None, tokenizer=None, true_id=0, false_id=1, max_length=512, device="cpu",
        sample_attack_tsv_path="fake/path.tsv", candidate_top_k=10,
        load_attack_tsv=fake_load_attack_tsv, build_candidate_sets=fake_build_candidate_sets,
        batch_size=8,
    )

    assert set(candidate_scores.keys()) == {"q1", "q2"}
    assert set(candidate_scores["q1"].keys()) == {"d1", "d2"}
    assert set(candidate_scores["q2"].keys()) == {"d3"}
    # 3 candidates total were scored in one batched call
    assert len(calls["texts"]) == 3
    assert set(candidate_sets.keys()) == {"q1", "q2"}
