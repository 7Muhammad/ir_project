from __future__ import annotations

import pandas as pd

from exp12lib.aggregation import example_balanced_aggregate, intervention_weighted_aggregate


def _row(attack, qid, docid, layer, group, combined, fwd=None, rev=None):
    return {
        "attack_name": attack, "qid": qid, "docid": docid, "layer": layer, "word_group": group,
        "forward_effect": fwd if fwd is not None else combined,
        "reverse_effect": rev if rev is not None else combined,
        "combined_effect": combined,
    }


def test_example_balanced_aggregate_does_not_let_long_queries_dominate():
    """
    Example A has 4 "content_matched" words in the group at layer 0: [0, 0, 0, 8].
        -> raw mean would be 2.0, but example-balanced averages them to 2.0
           WITHIN example A first (unavoidable single value: 2.0), then that
           2.0 counts once against example B.
    Example B has 1 "content_matched" word at layer 0: [4].
        -> per-example value: 4.0.
    Raw mean over all 5 raw rows = (0+0+0+8+4)/5 = 2.4.
    Example-balanced mean = mean(2.0, 4.0) = 3.0 -- example A's 4 words don't
    get 4x the weight of example B's 1 word.
    """
    df = pd.DataFrame([
        _row("atk", "q1", "d1", 0, "content_matched", 0.0),
        _row("atk", "q1", "d1", 0, "content_matched", 0.0),
        _row("atk", "q1", "d1", 0, "content_matched", 0.0),
        _row("atk", "q1", "d1", 0, "content_matched", 8.0),
        _row("atk", "q2", "d2", 0, "content_matched", 4.0),
    ])
    balanced = example_balanced_aggregate(df, ["layer", "word_group"])
    row = balanced[(balanced.layer == 0) & (balanced.word_group == "content_matched")].iloc[0]
    assert row["mean_combined"] == 3.0
    assert row["n_examples"] == 2
    assert row["n_word_interventions"] == 5

    weighted = intervention_weighted_aggregate(df, ["layer", "word_group"])
    wrow = weighted[(weighted.layer == 0) & (weighted.word_group == "content_matched")].iloc[0]
    assert wrow["mean_combined"] == 2.4  # the naive, word-count-biased number


def test_example_balanced_aggregate_separates_groups_and_layers():
    df = pd.DataFrame([
        _row("atk", "q1", "d1", 0, "content_matched", 1.0),
        _row("atk", "q1", "d1", 0, "stopword_matched", 5.0),
        _row("atk", "q1", "d1", 1, "content_matched", 9.0),
    ])
    balanced = example_balanced_aggregate(df, ["layer", "word_group"])
    assert len(balanced) == 3
    def val(layer, group):
        return balanced[(balanced.layer == layer) & (balanced.word_group == group)]["mean_combined"].iloc[0]
    assert val(0, "content_matched") == 1.0
    assert val(0, "stopword_matched") == 5.0
    assert val(1, "content_matched") == 9.0


def test_global_aggregate_pools_across_attacks():
    df = pd.DataFrame([
        _row("attackA", "q1", "d1", 0, "content_matched", 2.0),
        _row("attackB", "q2", "d2", 0, "content_matched", 6.0),
    ])
    per_attack = example_balanced_aggregate(df, ["attack_name", "layer", "word_group"])
    assert len(per_attack) == 2

    global_agg = example_balanced_aggregate(df, ["layer", "word_group"])
    assert len(global_agg) == 1
    assert global_agg.iloc[0]["mean_combined"] == 4.0
    assert global_agg.iloc[0]["n_examples"] == 2
