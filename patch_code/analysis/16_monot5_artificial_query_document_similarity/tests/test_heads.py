"""Head-level similarity: canonical head lists, pre-o_proj encoder slices, decoder per-head source split."""

from __future__ import annotations

import torch

from exp13lib.head_lists import ReceiverHead, SenderHead
from exp16lib.heads import EncoderHeadCapture, decoder_heads, encoder_heads, head_forward_batch
from exp16lib.inputs import EncodedSeq, collate
from exp16lib.pooling import pooled_cosine

CPU = torch.device("cpu")
ENC = [SenderHead(0, 1, "L0H1"), SenderHead(2, 3, "L2H3")]
DEC = [ReceiverHead(0, 2, "decoder_cross_attn", "L0-X-H2"), ReceiverHead(1, 0, "decoder_cross_attn", "L1-X-H0")]


def _pair(S=14, slots=(8, 9)):
    g = torch.Generator().manual_seed(9)
    ids = torch.randint(2, 100, (S,), generator=g).tolist()
    q = [1 if 2 <= i < 5 else 0 for i in range(S)]
    d = [1 if 6 <= i < 12 else 0 for i in range(S)]
    atk = EncodedSeq(ids, [1] * S, q, d)
    c_ids = [0 if i in slots else t for i, t in enumerate(ids)]
    c_att = [0 if i in slots else 1 for i in range(S)]
    return atk, EncodedSeq(c_ids, c_att, q, [a * b for a, b in zip(d, c_att)])


def test_canonical_head_lists():
    e, d = encoder_heads(), decoder_heads()
    assert len(e) == 18 and len(d) == 31
    assert all(h.component == "decoder_cross_attn" for h in d)
    assert [(h.layer, h.head_idx) for h in e] == sorted((h.layer, h.head_idx) for h in e)
    assert [(h.layer, h.head_idx) for h in d] == sorted((h.layer, h.head_idx) for h in d)
    assert {h.label for h in e} >= {"L10H0", "L9H6", "L8H11", "L11H3"}
    assert {h.label for h in d} >= {"L11-X-H3", "L8-X-H7"}


def test_encoder_head_is_pre_o_proj_slice(tiny_model):
    atk, _ = _pair()
    b = collate([atk], 0, CPU)
    attn_out = {}
    hs = [tiny_model.encoder.block[L].layer[0].SelfAttention.register_forward_hook(
        lambda m, a, o, _L=L: attn_out.__setitem__(_L, o[0])) for L in (0, 2)]
    with torch.no_grad(), EncoderHeadCapture(tiny_model.encoder, ENC, keep_inputs=True) as cap:
        cap.set_masks(b["query_mask"], b["doc_mask"])
        tiny_model.encoder(input_ids=b["input_ids"], attention_mask=b["attention_mask"])
    for h in hs:
        h.remove()
    dk = tiny_model.config.d_kv
    for L in (0, 2):
        o = tiny_model.encoder.block[L].layer[0].SelfAttention.o
        assert torch.allclose(o(cap.inputs[L]), attn_out[L], atol=1e-6)    # captured tensor is o's input
    exp = pooled_cosine(cap.inputs[2][..., 3 * dk:4 * dk], b["query_mask"], b["doc_mask"])
    assert torch.allclose(cap.cos["L2H3"], exp)


def test_decoder_head_split_matches_independent_recompute(tiny_model):
    atk, ctl = _pair()
    b = collate([atk, ctl], 0, CPU)
    enc, dec, err = head_forward_batch(tiny_model, b, ENC, DEC)
    assert err < 1e-6
    with torch.no_grad():
        out = tiny_model(input_ids=b["input_ids"], attention_mask=b["attention_mask"],
                         decoder_input_ids=torch.zeros(2, 1, dtype=torch.long), output_attentions=True)
        enc_out = out.encoder_last_hidden_state
        m = tiny_model.decoder.block[1].layer[1].EncDecAttention
        P = out.cross_attentions[1][:, 0, 0, :].double()                    # head 0, decoder step 0
        V = m.v(enc_out).view(2, -1, m.n_heads, m.key_value_proj_dim)[:, :, 0, :].double()
    zq = (P * b["query_mask"]).unsqueeze(-1).mul(V).sum(1)
    zd = (P * b["doc_mask"]).unsqueeze(-1).mul(V).sum(1)
    assert torch.allclose(dec["L1-X-H0"], torch.nn.functional.cosine_similarity(zq, zd, dim=-1), atol=1e-8)


def test_head_similarity_ignores_masked_slots(tiny_model):
    _, ctl = _pair()
    e1, d1, _ = head_forward_batch(tiny_model, collate([ctl], 0, CPU), ENC, DEC)
    ids = list(ctl.input_ids)
    ids[8], ids[9] = 61, 44
    ctl2 = EncodedSeq(ids, ctl.attention_mask, ctl.query_mask, ctl.doc_mask)
    e2, d2, _ = head_forward_batch(tiny_model, collate([ctl2], 0, CPU), ENC, DEC)
    for k in e1:
        assert torch.allclose(e1[k], e2[k], atol=1e-6)
    for k in d1:
        assert torch.allclose(d1[k], d2[k], atol=1e-6)
