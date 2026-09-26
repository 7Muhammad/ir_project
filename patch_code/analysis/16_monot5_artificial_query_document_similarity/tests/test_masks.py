"""Tests 6-9 on REAL prompts (monoT5 tokenizer + Exp 01 attack pairs): query span, doc span, attack inclusion, control exclusion."""

from __future__ import annotations

import pytest

from conftest import exp1_pair
from exp16lib.inputs import EncodingError, encode_attack_and_control, encode_clean

ATTACKS = ["relevant_start_5", "true_end_2", "bar_random_3", "relevance_start_1"]


def _tokens(tok, ids, mask):
    return [t for t, m in zip(tok.convert_ids_to_tokens(ids), mask) if m]


def test_06_query_span_is_query_text_only(tokenizer):
    p = exp1_pair("relevant_start_5", 0)
    s = encode_clean(tokenizer, p["query"], p["passage"], 512)
    qtok = _tokens(tokenizer, s.input_ids, s.query_mask)
    assert tokenizer.convert_tokens_to_string(qtok).strip() == p["query"].strip()
    toks = tokenizer.convert_ids_to_tokens(s.input_ids)
    q0 = s.query_mask.index(1)
    assert toks[:q0] == ["▁", "Query", ":"]            # template before the query is excluded
    q1 = q0 + sum(s.query_mask)
    assert toks[q1:q1 + 2] == ["▁Document", ":"]       # template after the query is excluded
    for bad in ("Query", "▁Document", "▁Relevan", "</s>"):
        assert bad not in qtok


def test_07_document_span_excludes_query_template_suffix(tokenizer):
    p = exp1_pair("relevant_start_5", 3)
    s = encode_clean(tokenizer, p["query"], p["passage"], 512)
    dtok = _tokens(tokenizer, s.input_ids, s.doc_mask)
    assert tokenizer.convert_tokens_to_string(dtok).strip() == tokenizer.decode(
        tokenizer.encode(p["passage"], add_special_tokens=False)).strip()
    toks = tokenizer.convert_ids_to_tokens(s.input_ids)
    d1 = max(i for i, m in enumerate(s.doc_mask) if m) + 1
    assert toks[d1:] == ["▁Relevan", "t", ":", "</s>"]
    assert not any(q and d for q, d in zip(s.query_mask, s.doc_mask))


@pytest.mark.parametrize("attack", ATTACKS)
def test_08_09_attack_tokens_included_control_slots_excluded(tokenizer, attack):
    p = exp1_pair(attack, 1)
    a, c, info = encode_attack_and_control(tokenizer, p["query"], p["passage"], p["attacked_passage"], 512)
    A = info["inserted_positions"]
    # 8: every injected position is active and inside the attacked document pool
    assert all(a.attention_mask[i] == 1 and a.doc_mask[i] == 1 for i in A) or info["alignment_boundary_shift"]
    assert a.n_doc_tokens == info["n_passage_tokens"] + info["n_inserted"]
    # 9: every insertion slot is masked in the control and outside its document pool
    assert all(c.attention_mask[i] == 0 and c.doc_mask[i] == 0 for i in A)
    assert c.n_doc_tokens == info["n_passage_tokens"] or info["alignment_boundary_shift"]
    # control doc pool = active doc positions only; pooled token ids == clean passage token ids
    clean = encode_clean(tokenizer, p["query"], p["passage"], 512)
    c_doc_ids = [t for t, m in zip(c.input_ids, c.doc_mask) if m]
    clean_doc_ids = [t for t, m in zip(clean.input_ids, clean.doc_mask) if m]
    assert c_doc_ids == clean_doc_ids
    # attack doc pool contains the injected token text
    atok = tokenizer.convert_tokens_to_string(_tokens(tokenizer, a.input_ids, a.doc_mask))
    assert attack.split("_")[0] in atok
    # query pool identical in attack and control and equal to the clean query pool
    assert a.query_mask == c.query_mask
    assert [t for t, m in zip(a.input_ids, a.query_mask) if m] == [t for t, m in zip(clean.input_ids, clean.query_mask) if m]


def test_no_silent_truncation(tokenizer):
    with pytest.raises(EncodingError):
        encode_clean(tokenizer, "q", "word " * 600, 512)
