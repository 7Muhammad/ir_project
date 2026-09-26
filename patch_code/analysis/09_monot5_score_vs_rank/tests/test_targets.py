"""Tests for exp9lib/targets.py."""

from __future__ import annotations

import csv

from exp9lib.targets import load_targets

FIELDS = ["qid", "docid", "rank", "bm25_score", "original_score",
          "control_score", "attack_score", "control_delta_vs_original",
          "attack_delta_vs_control", "attack_delta_vs_original"]


def test_load_targets_reads_all_rows_and_casts_scores(tmp_path):
    attack_dir = tmp_path / "fake_attack" / "scores"
    attack_dir.mkdir(parents=True)
    with open(attack_dir / "all_scores.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerow({
            "qid": "q1", "docid": "d1", "rank": "5", "bm25_score": "1.2",
            "original_score": "0.1", "control_score": "0.2", "attack_score": "0.9",
            "control_delta_vs_original": "0.1", "attack_delta_vs_control": "0.7",
            "attack_delta_vs_original": "0.8",
        })
        writer.writerow({
            "qid": "q1", "docid": "d2", "rank": "8", "bm25_score": "0.9",
            "original_score": "0.0", "control_score": "-0.1", "attack_score": "-0.2",
            "control_delta_vs_original": "-0.1", "attack_delta_vs_control": "-0.1",
            "attack_delta_vs_original": "-0.2",
        })

    targets = load_targets("fake_attack", tmp_path)
    assert len(targets) == 2
    assert targets[0]["qid"] == "q1"
    assert isinstance(targets[0]["control_score"], float)
    assert isinstance(targets[0]["attack_score"], float)
    assert targets[0]["control_score"] == 0.2
    assert targets[1]["attack_score"] == -0.2
