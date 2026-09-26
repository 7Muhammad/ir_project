"""Tests 1-5, 9, 10: checkpoint list, hook shapes/semantics, pad-slot and batch-padding exclusion."""

from __future__ import annotations

import torch

from exp16lib.checkpoints import CHECKPOINTS, checkpoint_names, checkpoint_table
from exp16lib.engine import forward_batch, run_sequences
from exp16lib.inputs import EncodedSeq, collate
from exp16lib.sanity import assert_hook_semantics, hook_semantics_report

CPU = torch.device("cpu")


def _seq(ids, attn, q, d):
    return EncodedSeq(list(ids), list(attn), list(q), list(d))


def _toy_pair(S=14, slots=(8, 9)):
    """Attack (all active) + padded control (slots masked); query 2..5, doc 7..11."""
    g = torch.Generator().manual_seed(1)
    ids = torch.randint(2, 100, (S,), generator=g).tolist()
    q = [1 if 2 <= i < 5 else 0 for i in range(S)]
    d = [1 if 6 <= i < 12 else 0 for i in range(S)]
    atk = _seq(ids, [1] * S, q, d)
    c_ids = [0 if i in slots else t for i, t in enumerate(ids)]
    c_att = [0 if i in slots else 1 for i in range(S)]
    ctl = _seq(c_ids, c_att, q, [dd * a for dd, a in zip(d, c_att)])
    return atk, ctl


def test_01_checkpoint_count():
    assert len(CHECKPOINTS) == 25
    assert len(checkpoint_table()) == 25
    assert len(set(CHECKPOINTS)) == 25


def test_02_checkpoint_order():
    expected = ["embedding"]
    for L in range(12):
        expected += [f"L{L:02d}_post_attn", f"L{L:02d}_post_mlp"]
    assert CHECKPOINTS == expected
    assert CHECKPOINTS[:3] == ["embedding", "L00_post_attn", "L00_post_mlp"]
    assert CHECKPOINTS[-2:] == ["L11_post_attn", "L11_post_mlp"]
    tab = checkpoint_table()
    assert [r["checkpoint_index"] for r in tab] == list(range(25))
    assert tab[0]["layer"] == -1 and tab[1]["sublayer"] == "attn" and tab[24] == {
        "checkpoint_index": 24, "checkpoint_name": "L11_post_mlp", "layer": 11, "sublayer": "mlp"}


def test_03_checkpoint_shapes(tiny_model):
    atk, ctl = _toy_pair()
    batch = collate([atk, ctl], 0, CPU)
    _, _, cap = forward_batch(tiny_model, batch, keep_states=True)
    assert list(cap.states) == checkpoint_names(3)
    for v in cap.states.values():
        assert tuple(v.shape) == (2, 14, tiny_model.config.d_model)


def test_04_05_post_attn_and_post_mlp_semantics(tiny_model):
    atk, ctl = _toy_pair()
    batch = collate([atk, ctl], 0, CPU)
    rep = assert_hook_semantics(tiny_model, batch, atol=1e-5, expected_checkpoints=7)
    # post-attention = residual AFTER the attention update (not the raw attention output)
    assert rep["post_attn_residual"] < 1e-5
    assert rep["post_attn_differs_from_raw_attention_output"] > 1e-3
    # post-MLP = residual after FFN update, carried into the next block, PRE final norm
    assert rep["post_mlp_residual"] < 1e-5 and rep["post_mlp_carried_forward"] < 1e-5
    assert rep["final_norm_of_last_post_mlp_vs_encoder_output"] < 1e-5
    assert rep["last_post_mlp_minus_encoder_output"] > 1e-3
    assert rep["embedding_vs_embed_tokens"] == 0.0


def test_semantics_checker_detects_wrong_hook(tiny_model, monkeypatch):
    """A capture that recorded the raw attention output would fail the report."""
    atk, _ = _toy_pair()
    rep = hook_semantics_report(tiny_model, collate([atk], 0, CPU))
    assert rep["post_attn_differs_from_raw_attention_output"] > 1e-3


def test_09_pad_control_slots_do_not_contribute(tiny_model):
    atk, ctl = _toy_pair()
    s1, _, _ = forward_batch(tiny_model, collate([ctl], 0, CPU))
    # change the token ids sitting in the masked slots: nothing may change
    ids2 = list(ctl.input_ids)
    ids2[8], ids2[9] = 77, 55
    ctl2 = _seq(ids2, ctl.attention_mask, ctl.query_mask, ctl.doc_mask)
    s2, _, _ = forward_batch(tiny_model, collate([ctl2], 0, CPU))
    assert torch.allclose(s1, s2, atol=1e-10)
    assert ctl.n_doc_tokens == atk.n_doc_tokens - 2
    # while including them (the attack) does change the similarity
    s3, _, _ = forward_batch(tiny_model, collate([atk], 0, CPU))
    assert not torch.allclose(s1, s3, atol=1e-4)


def test_10_batch_padding_never_contributes(tiny_model):
    atk, ctl = _toy_pair()
    short = _seq(atk.input_ids[:13], [1] * 13, atk.query_mask[:13], atk.doc_mask[:13])
    alone, _, _ = forward_batch(tiny_model, collate([short], 0, CPU))
    long_ = _seq(list(range(10, 30)), [1] * 20, [0] * 3 + [1] * 5 + [0] * 12, [0] * 9 + [1] * 8 + [0] * 3)
    joint, _, _ = forward_batch(tiny_model, collate([short, long_, ctl], 0, CPU))
    assert torch.allclose(alone[0], joint[0], atol=1e-6)
    # run_sequences (length sorting + batching) returns input order and same values
    S, _ = run_sequences(tiny_model, [long_, short, ctl], 0, CPU, batch_size=2)
    assert abs(S[1] - alone[0].numpy()).max() < 1e-6


def test_run_sequences_with_score(tiny_model):
    atk, ctl = _toy_pair()
    S, sc = run_sequences(tiny_model, [atk, ctl], 0, CPU, 2, with_score=True, true_id=5, false_id=7)
    assert S.shape == (2, 7) and sc.shape == (2,)
    with torch.no_grad():
        b = collate([atk], 0, CPU)
        lg = tiny_model(input_ids=b["input_ids"], attention_mask=b["attention_mask"],
                        decoder_input_ids=torch.zeros(1, 1, dtype=torch.long)).logits[0, 0]
    assert abs(float(lg[5] - lg[7]) - sc[0]) < 1e-5
