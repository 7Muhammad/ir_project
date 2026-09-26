"""I (attack alignment) and J (query span) on real Exp 01 pool examples + the real tokenizer."""

from __future__ import annotations

import pytest

from conftest import EXP1_ATTACKS

from exp15lib.positions import build_encodings_and_positions
from exp15lib.sampling import load_pool, sample_attack
from src.model_utils import build_monot5_input

CASES = [  # (attack, position, reps) — start/end/random, reps 1 and > 1
    ("relevant_start_1", "start", 1), ("relevant_start_5", "start", 5),
    ("true_end_1", "end", 1), ("true_end_4", "end", 4),
    ("bar_random_1", "random", 1), ("bar_random_3", "random", 3), ("important_random_5", "random", 5),
]


def _examples(attack, n=6):
    return sample_attack(load_pool(EXP1_ATTACKS, attack), attack, 42, n)


@pytest.mark.parametrize("attack,position,reps", CASES)
def test_I_attack_positions_capture_every_injected_token(tokenizer, attack, position, reps):
    token = attack.split("_")[0]
    for ex in _examples(attack):
        c, a, info = build_encodings_and_positions(tokenizer, ex["query"], ex["passage"], ex["attacked_passage"], 512)
        A = info["attack_source_positions"]
        a_ids = a["input_ids"][0].tolist()
        clean = tokenizer.encode(build_monot5_input(ex["query"], ex["passage"]))
        # removing A gives back the clean prompt, and A has exactly the extra length
        assert [t for i, t in enumerate(a_ids) if i not in set(A)] == clean
        assert len(A) == len(a_ids) - len(clean)
        # the injected text contains the attack word exactly `reps` times
        injected = tokenizer.decode([a_ids[i] for i in A]).replace(" ", "")
        assert injected.lower().count(token) == reps
        # padded control masks exactly A
        cm = c["attention_mask"][0].tolist()
        assert [i for i, v in enumerate(cm) if v == 0] == A
        if position == "random" and reps > 1:
            assert len(info["attack_spans"]) >= 1
        if position == "start" and not info["alignment_boundary_shift"]:
            assert A[0] == info["doc_span"][0]
        if position == "end":
            assert A[-1] == info["doc_span"][1] - 1


def test_I_random_multi_rep_is_scattered(tokenizer):
    spans = [len(build_encodings_and_positions(tokenizer, e["query"], e["passage"], e["attacked_passage"], 512)
                 [2]["attack_spans"]) for e in _examples("bar_random_3", 10)]
    assert max(spans) > 1


@pytest.mark.parametrize("attack", ["relevant_start_5", "true_end_2", "bar_random_3"])
def test_J_query_span_is_query_text_only(tokenizer, attack):
    for ex in _examples(attack):
        _, a, info = build_encodings_and_positions(tokenizer, ex["query"], ex["passage"], ex["attacked_passage"], 512)
        ids = a["input_ids"][0].tolist()
        Q = info["query_target_positions"]
        toks = tokenizer.convert_ids_to_tokens(ids)
        assert Q == list(range(Q[0], Q[-1] + 1))
        assert toks[:Q[0]] == ["▁", "Query", ":"]                       # leading template excluded
        assert toks[Q[-1] + 1:Q[-1] + 3] == ["▁Document", ":"]          # mid template excluded
        assert toks[-4:] == ["▁Relevan", "t", ":", "</s>"]              # tail template excluded
        assert tokenizer.decode([ids[q] for q in Q]).strip() == ex["query"].strip()
        assert max(Q) < info["doc_span"][0] and not set(Q) & set(info["attack_source_positions"])
        assert info["query_token_ids"] == [ids[q] for q in Q]
