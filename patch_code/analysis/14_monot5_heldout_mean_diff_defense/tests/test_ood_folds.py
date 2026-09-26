from __future__ import annotations

import pathlib

from src.attack_registry import AttackSpec

from exp14lib.data_pool import pool_round_robin
from exp14lib.ood_folds import FACTORS, build_all_folds, build_folds


def _spec(token, position, reps):
    return AttackSpec(
        attack_name=f"{token}_{position}_{reps}", token=token, position=position,
        repetitions=reps, run_name="bm25_19", path=pathlib.Path("/dev/null"),
    )


def _full_grid():
    tokens = ["relevant", "true", "false", "information", "bar", "important", "relevance"]
    positions = ["start", "end", "random"]
    reps = [1, 2, 3, 4, 5]
    return [_spec(t, p, r) for t in tokens for p in positions for r in reps]


def test_leave_one_token_out_has_seven_folds_and_correct_partition():
    """Test 15."""
    specs = _full_grid()
    folds = build_folds(specs, "token")
    assert len(folds) == 7
    all_names = {s.attack_name for s in specs}
    for fold in folds:
        seen, held = set(fold.seen_attacks), set(fold.held_out_attacks)
        assert seen.isdisjoint(held)
        assert seen | held == all_names
        assert len(held) == 3 * 5  # one token x 3 positions x 5 reps
        # every held-out attack really has the held-out token
        for name in fold.held_out_attacks:
            assert name.startswith(fold.held_out_value + "_")


def test_leave_one_position_out_has_three_folds_and_correct_partition():
    """Test 16."""
    specs = _full_grid()
    folds = build_folds(specs, "position")
    assert len(folds) == 3
    all_names = {s.attack_name for s in specs}
    for fold in folds:
        seen, held = set(fold.seen_attacks), set(fold.held_out_attacks)
        assert seen.isdisjoint(held)
        assert seen | held == all_names
        assert len(held) == 7 * 5  # 7 tokens x one position x 5 reps
        assert fold.held_out_value in {"start", "end", "random"}


def test_leave_one_repetition_out_has_five_folds_and_correct_partition():
    """Test 17."""
    specs = _full_grid()
    folds = build_folds(specs, "repetitions")
    assert len(folds) == 5
    all_names = {s.attack_name for s in specs}
    for fold in folds:
        seen, held = set(fold.seen_attacks), set(fold.held_out_attacks)
        assert seen.isdisjoint(held)
        assert seen | held == all_names
        assert len(held) == 7 * 3  # 7 tokens x 3 positions x one repetition count


def test_build_all_folds_covers_all_three_factors():
    specs = _full_grid()
    all_folds = build_all_folds(specs)
    assert set(all_folds.keys()) == set(FACTORS)
    assert len(all_folds["token"]) == 7
    assert len(all_folds["position"]) == 3
    assert len(all_folds["repetitions"]) == 5


def test_no_held_out_attack_examples_enter_the_pooled_training_set():
    """
    Test 18: simulates the exact filter-then-pool mechanism
    scripts/07_run_attack_ood.py uses for direction fitting, and confirms
    no example tagged with a held-out attack name ever appears in the
    pooled training set for that fold.
    """
    specs = _full_grid()
    folds = build_folds(specs, "token")
    fold = folds[0]  # held_out_value = "bar" (alphabetically first token)

    # Fake train_by_attack: 2 examples per attack, tagged with the attack name.
    train_by_attack = {
        s.attack_name: [{"qid": f"q{i}", "docid": f"d{i}", "attack": s.attack_name} for i in range(2)]
        for s in specs
    }

    seen_set = set(fold.seen_attacks)
    train_seen = {a: exs for a, exs in train_by_attack.items() if a in seen_set}
    pooled = pool_round_robin(train_seen, cap=None)

    held_out_set = set(fold.held_out_attacks)
    pooled_attacks = {ex["_source_attack"] for ex in pooled}
    assert pooled_attacks.isdisjoint(held_out_set)
    assert pooled_attacks == seen_set  # every seen attack contributed, no held-out attack did
