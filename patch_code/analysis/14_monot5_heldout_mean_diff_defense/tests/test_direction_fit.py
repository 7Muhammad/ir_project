from __future__ import annotations

import torch

from exp2lib.direction_fit import new_accumulator

import exp14lib.direction_fit as direction_fit_mod
from exp14lib.direction_fit import candidate_dir_keys, fit_directions_train_only
from exp14lib.head_lists import DecoderHead, EncoderHead


class _FakeModel:
    class _Config:
        num_heads = 4
        d_kv = 8
        num_layers = 1
        num_decoder_layers = 1

    config = _Config()


def test_direction_arithmetic_is_mean_attack_minus_mean_control():
    """Test 4: direction = mean(attack) - mean(control), verified on synthetic per-head activations."""
    model = _FakeModel()
    acc = new_accumulator(model)
    d_kv = model.config.d_kv
    attention_mask = torch.ones(1, 1)  # decoder-shaped: no pooling needed (seq_len=1)

    per_head_cache_control_1 = {("decoder_cross_attn", 0): torch.full((1, 1, model.config.num_heads * d_kv), 2.0)}
    per_head_cache_control_2 = {("decoder_cross_attn", 0): torch.full((1, 1, model.config.num_heads * d_kv), 4.0)}
    per_head_cache_attack_1 = {("decoder_cross_attn", 0): torch.full((1, 1, model.config.num_heads * d_kv), 10.0)}
    per_head_cache_attack_2 = {("decoder_cross_attn", 0): torch.full((1, 1, model.config.num_heads * d_kv), 12.0)}

    acc.add_example("control", {}, per_head_cache_control_1, attention_mask)
    acc.add_example("control", {}, per_head_cache_control_2, attention_mask)
    acc.add_example("attack", {}, per_head_cache_attack_1, attention_mask)
    acc.add_example("attack", {}, per_head_cache_attack_2, attention_mask)

    directions = acc.compute_directions()
    key = ("per_head", "decoder_cross_attn", 0, 0)
    assert key in directions
    # mean(control) = 3.0, mean(attack) = 11.0 -> direction = 8.0 (constant across the d_kv dims)
    assert torch.allclose(directions[key], torch.full((d_kv,), 8.0))


def test_fit_directions_train_only_is_a_pure_function_of_its_examples(monkeypatch):
    """
    Test 3: mean-diff direction fitting must be a pure function of the
    `train_examples` argument -- no hidden coupling to validation/test data.
    Verified by mocking the (expensive, model-requiring) activation-caching
    call and confirming: (a) it is invoked exactly twice (control+attack)
    per example in the given list, and (b) swapping in a different example
    list changes the accumulated counts accordingly.
    """
    calls = []

    def fake_build_encodings(tokenizer, query, passage, attacked_passage, max_length, device):
        class _AlignOK:
            status = "ok"
            n_inserted = 5
        control_enc = {"attention_mask": torch.ones(1, 3), "_role": "control", "_query": query}
        attack_enc = {"attention_mask": torch.ones(1, 3), "_role": "attack", "_query": query}
        return control_enc, attack_enc, _AlignOK()

    def fake_cache_activations(model, enc, true_id, false_id, device, n_enc, n_dec):
        calls.append((enc["_role"], enc["_query"]))
        value = 0.0 if enc["_role"] == "control" else 10.0
        d_kv = model.config.d_kv
        n_heads = model.config.num_heads
        whole = {}
        per_head = {
            ("encoder_self_attn", 0): torch.full((1, 3, n_heads * d_kv), value),
            ("decoder_cross_attn", 0): torch.full((1, 1, n_heads * d_kv), value),
        }
        score_control_or_attack = 0.0 if enc["_role"] == "control" else 1.0  # delta=1.0, above SKIP_EPSILON
        return whole, per_head, score_control_or_attack

    monkeypatch.setattr(direction_fit_mod, "build_padded_control_and_attack_encodings_general", fake_build_encodings)
    monkeypatch.setattr(direction_fit_mod, "cache_activations_for_direction", fake_cache_activations)

    model = _FakeModel()
    encoder_heads = [EncoderHead(layer=0, head_idx=0, label="L0H0", combined_effect_mean=0.1)]
    decoder_heads = [DecoderHead(layer=0, head_idx=0, component="decoder_cross_attn", label="L0-X-H0", combined_effect_mean=0.1)]

    train_examples_a = [
        {"query": "qa1", "passage": "p", "attacked_passage": "a"},
        {"query": "qa2", "passage": "p", "attacked_passage": "a"},
    ]
    directions_a, counts_a, stats_a = fit_directions_train_only(
        model, tokenizer=None, train_examples=train_examples_a,
        encoder_heads=encoder_heads, decoder_heads=decoder_heads,
        max_length=512, device=torch.device("cpu"), true_id=0, false_id=1,
    )
    assert stats_a["n_used"] == 2
    # exactly 2 calls (control+attack) per example, for exactly the given queries
    assert calls == [("control", "qa1"), ("attack", "qa1"), ("control", "qa2"), ("attack", "qa2")]

    calls.clear()
    train_examples_b = [{"query": "qb1", "passage": "p", "attacked_passage": "a"}]  # a DIFFERENT, smaller pool
    directions_b, counts_b, stats_b = fit_directions_train_only(
        model, tokenizer=None, train_examples=train_examples_b,
        encoder_heads=encoder_heads, decoder_heads=decoder_heads,
        max_length=512, device=torch.device("cpu"), true_id=0, false_id=1,
    )
    assert stats_b["n_used"] == 1
    assert calls == [("control", "qb1"), ("attack", "qb1")]

    key = ("per_head", "decoder_cross_attn", 0, 0)
    assert torch.allclose(directions_a[key], directions_b[key])  # same synthetic values -> same direction
    assert counts_a[key]["n_control"] == 2 and counts_b[key]["n_control"] == 1


def test_candidate_dir_keys_cover_encoder_and_decoder_heads():
    encoder_heads = [EncoderHead(layer=1, head_idx=2, label="L1H2", combined_effect_mean=0.1)]
    decoder_heads = [DecoderHead(layer=3, head_idx=4, component="decoder_cross_attn", label="L3-X-H4", combined_effect_mean=0.1)]
    keys = candidate_dir_keys(encoder_heads, decoder_heads)
    assert ("per_head", "encoder_self_attn", 1, 2) in keys
    assert ("per_head", "decoder_cross_attn", 3, 4) in keys
    assert len(keys) == 2
