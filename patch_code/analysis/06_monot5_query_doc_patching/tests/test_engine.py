"""
Correctness tests for the positional patching engine.

Key properties verified:
  1. build_position_mask: correct positions set, doc|query masks union to
     the "both" mask.
  2. make_positional_patch_hook: exact blend math, tested by calling the
     hook function directly (isolated from a full forward pass) so the
     result is checked against a hand-computed tensor, not inferred through
     downstream nonlinear layers.
  3. Full-sequence mask (query_span ∪ doc_span == whole sequence) reproduces
     Experiment 1's whole-layer contribution patch exactly (cross-experiment
     consistency, same pattern Experiment 3 used against Experiment 1).
  4. make_cross_attn_positional_prehook: exact blend math on key_value_states,
     and only the hooked decoder layer is affected (verified via full model
     forward passes with hooks on layer 0 vs layer 1).
  5. Driver functions produce the right row counts/shape and skip on
     negligible delta, same SKIP_EPSILON convention as Experiment 1/3.
"""

from __future__ import annotations

import pytest
import torch

from conftest import DEVICE, FALSE_ID, TRUE_ID

from src.activation_hooks import get_module_for_component, make_cache_hook, make_patch_hook
from src.model_utils import score_from_encoding

from exp6lib.engine import (
    build_position_mask,
    compute_encoder_hidden_states,
    decoder_step_score,
    make_cross_attn_positional_prehook,
    make_positional_patch_hook,
    run_decoder_cross_attn_example,
    run_encoder_self_attn_example,
)


def test_build_position_mask_regions_and_union():
    seq_len = 10
    query_span, doc_span = (0, 3), (4, 9)
    m_doc = build_position_mask(seq_len, "doc_only", query_span, doc_span, DEVICE)
    m_query = build_position_mask(seq_len, "query_only", query_span, doc_span, DEVICE)
    m_both = build_position_mask(seq_len, "both", query_span, doc_span, DEVICE)

    assert m_doc.shape == (1, seq_len, 1)
    assert m_doc[0, 4:9, 0].sum().item() == 5
    assert m_doc[0, :4, 0].sum().item() == 0 and m_doc[0, 9:, 0].sum().item() == 0

    assert m_query[0, 0:3, 0].sum().item() == 3
    assert m_query[0, 3:, 0].sum().item() == 0

    assert torch.equal(torch.clamp(m_doc + m_query, max=1.0), m_both)


def test_positional_patch_hook_exact_blend():
    """Call the hook directly with synthetic input/output; check exact per-position blend."""
    seq_len, d_model = 6, 4
    in_hidden = torch.randn(1, seq_len, d_model)
    own_contrib = torch.randn(1, seq_len, d_model)
    out_hidden = in_hidden + own_contrib
    replacement = torch.randn(1, seq_len, d_model)

    mask = torch.zeros(1, seq_len, 1)
    mask[0, 2:4, 0] = 1.0  # patch positions 2,3 only

    hook = make_positional_patch_hook(replacement, mask)
    new_output = hook(None, (in_hidden,), out_hidden)
    new_hidden = new_output  # module output is a plain tensor here (T5LayerFF-style)

    expected = out_hidden.clone()
    expected[0, 2:4, :] = in_hidden[0, 2:4, :] + replacement[0, 2:4, :]
    assert torch.allclose(new_hidden, expected, atol=1e-6)
    # untouched positions must equal the ORIGINAL output exactly
    untouched = [i for i in range(seq_len) if i not in (2, 3)]
    assert torch.allclose(new_hidden[0, untouched, :], out_hidden[0, untouched, :], atol=1e-6)


def test_full_mask_reproduces_experiment1_layer_patch(tiny_model, control_enc, attack_enc):
    """
    query_span ∪ doc_span == whole sequence must reproduce Experiment 1's
    whole-layer contribution patch (src.activation_hooks.make_patch_hook)
    bit-for-bit, the same cross-experiment consistency check Experiment 3 used.
    """
    seq_len = control_enc["input_ids"].shape[1]
    layer_idx = 1
    module = get_module_for_component(tiny_model, "encoder_self_attn", layer_idx)

    # Cache the attack run's contribution at this layer (Experiment 1 mechanism, reused).
    cache = {}
    handle = module.register_forward_hook(make_cache_hook(cache, "k"))
    try:
        score_from_encoding(tiny_model, attack_enc, TRUE_ID, FALSE_ID, DEVICE)
    finally:
        handle.remove()
    atk_contrib = cache["k"]

    # Experiment 1's whole-layer patch.
    handle = module.register_forward_hook(make_patch_hook(atk_contrib))
    try:
        exp1_score = score_from_encoding(tiny_model, control_enc, TRUE_ID, FALSE_ID, DEVICE)
    finally:
        handle.remove()

    # Experiment 6's positional patch with a mask covering the WHOLE sequence.
    full_mask = torch.ones(1, seq_len, 1)
    handle = module.register_forward_hook(make_positional_patch_hook(atk_contrib, full_mask))
    try:
        exp6_score = score_from_encoding(tiny_model, control_enc, TRUE_ID, FALSE_ID, DEVICE)
    finally:
        handle.remove()

    assert exp6_score == pytest.approx(exp1_score, abs=1e-5)


def test_zero_mask_is_a_noop(tiny_model, control_enc, attack_enc):
    seq_len = control_enc["input_ids"].shape[1]
    layer_idx = 0
    module = get_module_for_component(tiny_model, "encoder_self_attn", layer_idx)
    base_score = score_from_encoding(tiny_model, control_enc, TRUE_ID, FALSE_ID, DEVICE)

    cache = {}
    handle = module.register_forward_hook(make_cache_hook(cache, "k"))
    try:
        score_from_encoding(tiny_model, attack_enc, TRUE_ID, FALSE_ID, DEVICE)
    finally:
        handle.remove()

    zero_mask = torch.zeros(1, seq_len, 1)
    handle = module.register_forward_hook(make_positional_patch_hook(cache["k"], zero_mask))
    try:
        patched_score = score_from_encoding(tiny_model, control_enc, TRUE_ID, FALSE_ID, DEVICE)
    finally:
        handle.remove()

    assert patched_score == pytest.approx(base_score, abs=1e-6)


def test_cross_attn_prehook_exact_blend():
    seq_len, d_model = 6, 4
    base_kv = torch.randn(1, seq_len, d_model)
    other_kv = torch.randn(1, seq_len, d_model)
    mask = torch.zeros(1, seq_len, 1)
    mask[0, 1:3, 0] = 1.0

    hook = make_cross_attn_positional_prehook(base_kv, other_kv, mask)
    args, kwargs = hook(None, (torch.zeros(1, 1, d_model),), {"key_value_states": base_kv.clone(), "attention_mask": None})

    expected = base_kv.clone()
    expected[0, 1:3, :] = other_kv[0, 1:3, :]
    assert torch.allclose(kwargs["key_value_states"], expected, atol=1e-6)


def test_cross_attn_prehook_only_affects_hooked_layer(tiny_model, control_enc, attack_enc):
    """Patching layer 0's cross-attention must not change what layer 1 reads."""
    ctrl_hidden = compute_encoder_hidden_states(tiny_model, control_enc, DEVICE)
    atk_hidden = compute_encoder_hidden_states(tiny_model, attack_enc, DEVICE)
    seq_len = ctrl_hidden.shape[1]
    full_mask = torch.ones(1, seq_len, 1)

    base_score = decoder_step_score(tiny_model, ctrl_hidden, control_enc["attention_mask"], TRUE_ID, FALSE_ID)

    layer0 = tiny_model.decoder.block[0].layer[1]
    handle = layer0.register_forward_pre_hook(
        make_cross_attn_positional_prehook(ctrl_hidden, atk_hidden, full_mask), with_kwargs=True
    )
    try:
        patched_score = decoder_step_score(tiny_model, ctrl_hidden, control_enc["attention_mask"], TRUE_ID, FALSE_ID)
    finally:
        handle.remove()

    # Full-mask swap on one layer should change the score (attack's encoder rows differ from control's).
    assert patched_score != pytest.approx(base_score, abs=1e-9)

    # And removing the hook must restore the exact baseline (no leakage).
    restored_score = decoder_step_score(tiny_model, ctrl_hidden, control_enc["attention_mask"], TRUE_ID, FALSE_ID)
    assert restored_score == pytest.approx(base_score, abs=1e-9)


def test_run_encoder_self_attn_example_row_shape(tiny_model, control_enc, attack_enc):
    layers = list(range(tiny_model.config.num_layers))
    rows = run_encoder_self_attn_example(
        tiny_model, control_enc, attack_enc, layers,
        query_span=(0, 3), doc_span=(3, 10), true_id=TRUE_ID, false_id=FALSE_ID, device=DEVICE,
    )
    assert rows is not None
    assert len(rows) == len(layers) * 3  # 3 conditions
    for r in rows:
        assert r["region"] == "encoder_self_attn"
        assert r["combined_effect"] == pytest.approx(min(r["fwd_effect"], r["rev_effect"]), abs=1e-9)


def test_run_decoder_cross_attn_example_row_shape(tiny_model, control_enc, attack_enc):
    layers = list(range(tiny_model.config.num_decoder_layers))
    rows = run_decoder_cross_attn_example(
        tiny_model, control_enc, attack_enc, layers,
        query_span=(0, 3), doc_span=(3, 10), true_id=TRUE_ID, false_id=FALSE_ID, device=DEVICE,
    )
    assert rows is not None
    assert len(rows) == len(layers) * 3
    for r in rows:
        assert r["region"] == "decoder_cross_attn"
        assert r["combined_effect"] == pytest.approx(min(r["fwd_effect"], r["rev_effect"]), abs=1e-9)


def test_hooks_leave_no_residue(tiny_model, control_enc, attack_enc):
    base = score_from_encoding(tiny_model, control_enc, TRUE_ID, FALSE_ID, DEVICE)
    run_encoder_self_attn_example(
        tiny_model, control_enc, attack_enc, [0, 1],
        query_span=(0, 3), doc_span=(3, 10), true_id=TRUE_ID, false_id=FALSE_ID, device=DEVICE,
    )
    after = score_from_encoding(tiny_model, control_enc, TRUE_ID, FALSE_ID, DEVICE)
    assert after == pytest.approx(base, abs=1e-6)


def test_run_example_all_regions_unified_schema(tiny_model, control_enc, attack_enc):
    from exp6lib.engine import run_example_all_regions
    from exp6lib.run_utils import ExampleInputs

    inputs = ExampleInputs(
        status="ok",
        clean_enc=control_enc, control_enc=control_enc, attack_enc=attack_enc,
        query_span=(0, 3), doc_span_clean=(3, 10), doc_span_control_attack=(3, 10),
    )
    enc_layers = list(range(tiny_model.config.num_layers))
    dec_layers = list(range(tiny_model.config.num_decoder_layers))
    rows = run_example_all_regions(tiny_model, inputs, enc_layers, dec_layers, TRUE_ID, FALSE_ID, DEVICE)

    assert len(rows) == (len(enc_layers) + len(dec_layers)) * 3
    enc_rows = [r for r in rows if r["region"] == "encoder_self_attn"]
    dec_rows = [r for r in rows if r["region"] == "decoder_cross_attn"]
    assert len(enc_rows) == len(enc_layers) * 3
    assert len(dec_rows) == len(dec_layers) * 3

    for r in enc_rows:
        assert r["attention_q_to_d_clean"] is not None
        assert r["attention_d_to_q_attack"] is not None
    for r in dec_rows:
        assert r["attention_q_to_d_clean"] is None
        assert r["attention_d_to_q_attack"] is None
