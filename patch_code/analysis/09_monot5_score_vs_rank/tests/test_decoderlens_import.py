"""Tests for exp9lib/decoderlens_import.py — the file-path loader that
avoids the `src` package name collision between Experiment 1 and
DecoderLens (both have a top-level package literally named `src`)."""

from __future__ import annotations

import csv
import pathlib

from conftest import DECODERLENS_DIR

from exp9lib.decoderlens_import import load_decoderlens_ranking

TSV_HEADER = ["qid", "query", "docno", "score", "rank", "text", "text_0"]


def _write_tsv(path: pathlib.Path, rows: list) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh, delimiter="\t")
        writer.writerow(TSV_HEADER)
        writer.writerows(rows)


def test_load_decoderlens_ranking_returns_working_functions(tmp_path):
    Candidate, load_attack_tsv, build_candidate_sets, compute_rank = (
        load_decoderlens_ranking(DECODERLENS_DIR)
    )

    tsv_path = tmp_path / "fake_attack.gz.tsv"
    _write_tsv(tsv_path, [
        ["q1", "what is x", "d1", "1.0", "1", "attacked passage 1", "clean passage 1"],
        ["q1", "what is x", "d2", "0.9", "2", "attacked passage 2", "clean passage 2"],
        ["q1", "what is x", "d3", "0.8", "3", "attacked passage 3", "clean passage 3"],
    ])

    records = load_attack_tsv(tsv_path)
    assert len(records) == 3
    assert all(isinstance(r, Candidate) for r in records)
    assert records[0].passage == "clean passage 1"
    assert records[0].attacked_passage == "attacked passage 1"

    candidate_sets = build_candidate_sets(records, candidate_top_k=2)
    assert list(candidate_sets.keys()) == ["q1"]
    assert [c.docid for c in candidate_sets["q1"]] == ["d1", "d2"]  # top-2 by rank

    scores = {"d1": 5.0, "d2": 3.0, "d3": 1.0}
    assert compute_rank(scores, "d2", target_score=5.5) == 1  # now beats d1
    assert compute_rank(scores, "d3", target_score=1.0) == 3  # unchanged, still last
    assert compute_rank(scores, "d3", target_score=10.0) == 1  # jumps to first


def test_no_src_package_collision_after_loading_decoderlens():
    """After loading DecoderLens's code, Experiment 1's `src` package must
    still resolve correctly — this is the regression the file-path loader
    exists to prevent."""
    load_decoderlens_ranking(DECODERLENS_DIR)

    import src.model_utils  # noqa: E402
    assert hasattr(src.model_utils, "load_monot5")
    assert hasattr(src.model_utils, "score_from_encoding")

    # DecoderLens's own modules must NOT have leaked into sys.modules under
    # the shared `src` name (they were loaded under private names instead).
    import sys
    assert "src.ranking" not in sys.modules
    assert "src.data_loading" not in sys.modules
