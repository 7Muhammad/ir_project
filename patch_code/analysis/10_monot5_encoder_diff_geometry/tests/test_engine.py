"""
Engine correctness: encoder_hidden_states shape/layer plumbing (synthetic
ids, tiny model) and compute_example_diffs end-to-end wiring (real
tokenizer + real example text, so tagging's span probes have real text to
match against, with a full-vocab-but-otherwise-tiny model so real token ids
are valid embedding indices).
"""

from __future__ import annotations

import pytest
import torch
from transformers import T5Config, T5ForConditionalGeneration

from src.model_utils import build_padded_control_and_attack_encodings_general
from exp10lib.engine import compute_example_diffs, encoder_hidden_states
from exp10lib.run_utils import ExampleInputs
from conftest import DEVICE


def test_encoder_hidden_states_shapes_and_layers(tiny_model, control_enc):
    layers = [0, 1, 2]
    hs = encoder_hidden_states(tiny_model, control_enc, layers)
    assert set(hs.keys()) == set(layers)
    seq_len = control_enc["input_ids"].shape[1]
    for L in layers:
        assert hs[L].shape == (seq_len, tiny_model.config.d_model)


def test_encoder_hidden_states_differ_by_layer(tiny_model, control_enc):
    """Different depths should (generically) give different representations."""
    hs = encoder_hidden_states(tiny_model, control_enc, [0, 2])
    assert not (hs[0] == hs[2]).all()


@pytest.fixture(scope="module")
def tiny_full_vocab_model(real_tokenizer):
    torch.manual_seed(0)
    cfg = T5Config(
        vocab_size=len(real_tokenizer), d_model=32, d_kv=8, num_heads=4, d_ff=64,
        num_layers=12, num_decoder_layers=1, decoder_start_token_id=0,
        dropout_rate=0.0,
    )
    model = T5ForConditionalGeneration(cfg)
    model.eval()
    return model


def test_compute_example_diffs_end_to_end(tiny_full_vocab_model, real_tokenizer, real_example):
    control_enc, attack_enc, align_result = build_padded_control_and_attack_encodings_general(
        tokenizer=real_tokenizer, query=real_example["query"], passage=real_example["passage"],
        attacked_passage=real_example["attacked_passage"], max_length=512, device=DEVICE,
    )
    assert align_result.status == "ok"
    seq_len = attack_enc["input_ids"].shape[1]
    inputs = ExampleInputs(status="ok", control_enc=control_enc, attack_enc=attack_enc,
                            align_result=align_result, seq_len=seq_len)

    layers = [9, 10, 11]
    status, reason, diffs, tags = compute_example_diffs(
        tiny_full_vocab_model, real_tokenizer, real_example["query"], real_example["passage"], inputs, layers,
    )
    assert status == "ok", reason
    assert len(tags) == seq_len
    for L in layers:
        assert diffs[L].shape == (seq_len, 32)
        # attack and control differ at least at the padded/inserted positions
        # (real content vs. pad embedding) even with random untrained weights.
        assert not (diffs[L] == 0).all()
