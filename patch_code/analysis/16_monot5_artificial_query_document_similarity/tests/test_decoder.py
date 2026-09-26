"""Decoder cross-attention message split: naming, exact decomposition, attention mass, masked-slot invariance."""

from __future__ import annotations

import torch

from exp16lib.decoder import DECODER_CHECKPOINTS, FIELDS, CrossAttnMessageCapture, decoder_forward_batch
from exp16lib.inputs import EncodedSeq, collate

CPU = torch.device("cpu")


def _pair(S=14, slots=(8, 9)):
    g = torch.Generator().manual_seed(5)
    ids = torch.randint(2, 100, (S,), generator=g).tolist()
    q = [1 if 2 <= i < 5 else 0 for i in range(S)]
    d = [1 if 6 <= i < 12 else 0 for i in range(S)]
    atk = EncodedSeq(ids, [1] * S, q, d)
    c_ids = [0 if i in slots else t for i, t in enumerate(ids)]
    c_att = [0 if i in slots else 1 for i in range(S)]
    ctl = EncodedSeq(c_ids, c_att, q, [a * b for a, b in zip(d, c_att)])
    return atk, ctl


def test_decoder_checkpoints():
    assert len(DECODER_CHECKPOINTS) == 12
    assert DECODER_CHECKPOINTS[0] == "D00_cross_attn" and DECODER_CHECKPOINTS[-1] == "D11_cross_attn"


def test_message_decomposition_exact_and_mass(tiny_model):
    atk, ctl = _pair()
    batch = collate([atk, ctl, EncodedSeq(atk.input_ids[:12], [1] * 12, atk.query_mask[:12], atk.doc_mask[:12])], 0, CPU)
    res, scores, err = decoder_forward_batch(tiny_model, batch, 5, 7)
    assert err < 1e-6                                    # relative: m_query + m_doc + m_rest == attention output
    for f in FIELDS:
        assert res[f].shape == (3, tiny_model.config.num_decoder_layers)
    assert ((res["mass_query"] + res["mass_doc"]) <= 1 + 1e-9).all()
    assert (res["msg_cos"].abs() <= 1 + 1e-9).all()
    # score equals an ordinary forward
    with torch.no_grad():
        lg = tiny_model(input_ids=batch["input_ids"][:1], attention_mask=batch["attention_mask"][:1],
                        decoder_input_ids=torch.zeros(1, 1, dtype=torch.long)).logits[0, 0]
    assert abs(float(lg[5] - lg[7]) - float(scores[0])) < 1e-5


def test_masked_slots_and_batch_padding_carry_no_message(tiny_model):
    atk, ctl = _pair()
    r1, _, _ = decoder_forward_batch(tiny_model, collate([ctl], 0, CPU), 5, 7)
    ids2 = list(ctl.input_ids)
    ids2[8], ids2[9] = 71, 33                             # change masked-slot ids: nothing may change
    ctl2 = EncodedSeq(ids2, ctl.attention_mask, ctl.query_mask, ctl.doc_mask)
    long_ = EncodedSeq(list(range(10, 30)), [1] * 20, [0] * 3 + [1] * 5 + [0] * 12, [0] * 9 + [1] * 8 + [0] * 3)
    r2, _, _ = decoder_forward_batch(tiny_model, collate([ctl2, long_], 0, CPU), 5, 7)
    for f in FIELDS:
        assert torch.allclose(r1[f][0], r2[f][0], atol=1e-6)
    ra, _, _ = decoder_forward_batch(tiny_model, collate([atk], 0, CPU), 5, 7)
    assert not torch.allclose(ra["msg_cos"][0], r1["msg_cos"][0], atol=1e-4)


def test_capture_requires_attention_weights(tiny_model):
    atk, _ = _pair()
    b = collate([atk], 0, CPU)
    import pytest
    with CrossAttnMessageCapture(tiny_model.decoder) as cap:
        cap.set_masks(b["query_mask"], b["doc_mask"])
        with pytest.raises(RuntimeError):
            with torch.no_grad():
                tiny_model(input_ids=b["input_ids"], attention_mask=b["attention_mask"],
                           decoder_input_ids=torch.zeros(1, 1, dtype=torch.long), use_cache=False)
