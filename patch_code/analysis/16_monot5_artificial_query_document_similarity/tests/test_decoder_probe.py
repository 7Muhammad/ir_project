"""Decoder query-only vs doc-only probe: checkpoints, trivial pre-cross-attn identity, span restriction, hook semantics."""

from __future__ import annotations

import torch
from transformers.modeling_outputs import BaseModelOutput

from exp16lib.decoder_probe import DECODER_PROBE_CHECKPOINTS, DecoderStateCapture, decoder_probe_batch
from exp16lib.inputs import EncodedSeq, collate

CPU = torch.device("cpu")


def _batch():
    g = torch.Generator().manual_seed(11)
    S = 14
    ids = torch.randint(2, 100, (S,), generator=g).tolist()
    q = [1 if 2 <= i < 5 else 0 for i in range(S)]
    d = [1 if 6 <= i < 12 else 0 for i in range(S)]
    c_att = [0 if i in (8, 9) else 1 for i in range(S)]
    a = EncodedSeq(ids, [1] * S, q, d)
    c = EncodedSeq([0 if i in (8, 9) else t for i, t in enumerate(ids)], c_att, q, [x * y for x, y in zip(d, c_att)])
    return collate([a, c], 0, CPU)


def test_checkpoint_names():
    assert len(DECODER_PROBE_CHECKPOINTS) == 37
    assert DECODER_PROBE_CHECKPOINTS[:4] == ["dec_embedding", "D00_post_self_attn", "D00_post_cross_attn", "D00_post_mlp"]
    assert DECODER_PROBE_CHECKPOINTS[-1] == "D11_post_mlp"


def test_pre_cross_attention_checkpoints_identical(tiny_model):
    sims, ex = decoder_probe_batch(tiny_model, _batch(), keep_states=True)
    assert sims.shape == (2, 1 + 3 * tiny_model.config.num_decoder_layers)
    assert torch.allclose(sims[:, :2], torch.ones(2, 2, dtype=torch.float64), atol=1e-12)
    assert not torch.allclose(sims[:, 2], torch.ones(2, dtype=torch.float64))   # first cross-attn separates them


def test_query_only_run_ignores_non_query_encoder_states(tiny_model):
    b = _batch()
    _, ex = decoder_probe_batch(tiny_model, b, keep_states=True)
    enc = ex["enc"].clone()
    enc[b["query_mask"] == 0] += 5.0 * torch.randn_like(enc[b["query_mask"] == 0])
    dec = torch.zeros(2, 1, dtype=torch.long)
    with torch.no_grad(), DecoderStateCapture(tiny_model.decoder) as cap:
        tiny_model(encoder_outputs=BaseModelOutput(last_hidden_state=enc), attention_mask=b["query_mask"],
                   decoder_input_ids=dec, use_cache=False)
        H = cap.stacked()
    assert torch.allclose(H, ex["hq"], atol=1e-5)


def test_hook_semantics_carry_forward_and_final_norm(tiny_model):
    b = _batch()
    _, ex = decoder_probe_batch(tiny_model, b, keep_states=True)
    enc2 = torch.cat([ex["enc"], ex["enc"]], 0)
    with torch.no_grad():
        out = tiny_model(encoder_outputs=BaseModelOutput(last_hidden_state=enc2),
                         attention_mask=torch.cat([b["query_mask"], b["doc_mask"]], 0),
                         decoder_input_ids=torch.zeros(4, 1, dtype=torch.long), output_hidden_states=True, use_cache=False)
    hs = out.decoder_hidden_states                     # input of each block, then final-normed output
    H = torch.cat([ex["hq"], ex["hd"]], 0)
    n = tiny_model.config.num_decoder_layers
    assert torch.allclose(H[:, 0], hs[0][:, 0], atol=1e-6)                           # embedding
    for L in range(1, n):
        assert torch.allclose(H[:, 3 * L], hs[L][:, 0], atol=1e-5)                   # post_mlp[L-1] == block_in[L]
    fn = tiny_model.decoder.final_layer_norm
    assert torch.allclose(fn(H[:, -1]), hs[-1][:, 0], atol=1e-5)                     # last post_mlp is PRE final norm
    assert not torch.allclose(H[:, -1], hs[-1][:, 0], atol=1e-3)
