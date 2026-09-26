"""
Correctness tests for Experiment 11's encoder per-head patching engine.

The important properties (encoder-side analog of Experiment 3's tests):
  1. The batched per-row head hook equals patching one head at a time.
  2. Replacing ALL head slices at the .o input reproduces Experiment 1's
     layer-level `encoder_self_attn` contribution patch (consistency across
     experiments -- same check exp6's report ran for positional patching,
     now for head patching).
  3. Zeroing all head slices equals a zero-contribution layer patch
     (W_o has no bias).
  4. run_grid_example's effect math is internally consistent, and
     score_ablated_mean == score_patched_rev on the attack base.
"""

from __future__ import annotations

import pytest
import torch

from conftest import DEVICE, FALSE_ID, TRUE_ID

from src.activation_hooks import get_module_for_component, make_cache_hook, make_patch_hook
from src.model_utils import score_from_encoding

from exp11lib.engine import cache_head_inputs, decoder_pass_scores_batched, head_scores_for_layer, run_grid_example
from exp11lib.head_hooks import (
    block_diag_head_mask,
    enumerate_layers,
    get_o_proj,
    head_geometry,
    make_full_replace_pre_hook,
    slot_key,
)


def test_batched_head_hook_equals_one_head_at_a_time(tiny_model, control_enc, attack_enc):
    """Row h of the batched pass must equal a single-row pass patching only head h."""
    layers = enumerate_layers(tiny_model)
    n_heads, d_kv, _ = head_geometry(tiny_model)
    head_mask = block_diag_head_mask(n_heads, d_kv, DEVICE)

    atk_cache, _ = cache_head_inputs(tiny_model, attack_enc, DEVICE, layers, TRUE_ID, FALSE_ID)

    layer = 1
    repl = atk_cache[slot_key(layer)]
    batched = head_scores_for_layer(
        tiny_model, control_enc, DEVICE, layer, repl, head_mask, TRUE_ID, FALSE_ID
    )

    o_proj = get_o_proj(tiny_model, layer)
    for h in range(n_heads):
        def one_head_hook(module, args, h=h):
            hid = args[0].clone()
            hid[:, :, h * d_kv:(h + 1) * d_kv] = repl[:, :, h * d_kv:(h + 1) * d_kv]
            return (hid,) + args[1:]

        handle = o_proj.register_forward_pre_hook(one_head_hook)
        try:
            single = score_from_encoding(tiny_model, control_enc, TRUE_ID, FALSE_ID, DEVICE)
        finally:
            handle.remove()
        assert batched[h] == pytest.approx(single, abs=1e-5), f"head {h} mismatch"

    base = score_from_encoding(tiny_model, control_enc, TRUE_ID, FALSE_ID, DEVICE)
    assert any(abs(s - base) > 1e-7 for s in batched)


@pytest.mark.parametrize("layer", [0, 1, 2])
def test_all_heads_patch_equals_exp1_layer_patch(tiny_model, control_enc, attack_enc, layer):
    """
    Replacing every head slice at the .o input with the attack run's slices
    must equal Experiment 1's layer-level `encoder_self_attn` contribution
    patch of the same layer (make_cache_hook / make_patch_hook).
    """
    layers = enumerate_layers(tiny_model)
    atk_cache, _ = cache_head_inputs(tiny_model, attack_enc, DEVICE, layers, TRUE_ID, FALSE_ID)

    # Head-level: replace the full .o input on the control base run.
    handle = get_o_proj(tiny_model, layer).register_forward_pre_hook(
        make_full_replace_pre_hook(atk_cache[slot_key(layer)])
    )
    try:
        head_level = score_from_encoding(tiny_model, control_enc, TRUE_ID, FALSE_ID, DEVICE)
    finally:
        handle.remove()

    # Experiment-1 layer level: cache the attack contribution of the whole
    # encoder_self_attn sub-layer, then patch it into the control run.
    module = get_module_for_component(tiny_model, "encoder_self_attn", layer)
    contrib_cache = {}
    handle = module.register_forward_hook(make_cache_hook(contrib_cache, "c"))
    try:
        score_from_encoding(tiny_model, attack_enc, TRUE_ID, FALSE_ID, DEVICE)
    finally:
        handle.remove()

    handle = module.register_forward_hook(make_patch_hook(contrib_cache["c"]))
    try:
        layer_level = score_from_encoding(tiny_model, control_enc, TRUE_ID, FALSE_ID, DEVICE)
    finally:
        handle.remove()

    assert head_level == pytest.approx(layer_level, abs=1e-4)


def test_zero_all_heads_equals_zero_contribution(tiny_model, attack_enc):
    """Zeroing every head slice must equal an Experiment-1 zero-contribution patch."""
    n_heads, d_kv, inner = head_geometry(tiny_model)
    layer = 0

    handle = get_o_proj(tiny_model, layer).register_forward_pre_hook(
        make_full_replace_pre_hook(torch.zeros(1, 1, inner))
    )
    try:
        head_level = score_from_encoding(tiny_model, attack_enc, TRUE_ID, FALSE_ID, DEVICE)
    finally:
        handle.remove()

    module = get_module_for_component(tiny_model, "encoder_self_attn", layer)
    d_model = tiny_model.config.d_model
    handle = module.register_forward_hook(make_patch_hook(torch.zeros(1, 1, d_model)))
    try:
        layer_level = score_from_encoding(tiny_model, attack_enc, TRUE_ID, FALSE_ID, DEVICE)
    finally:
        handle.remove()

    assert head_level == pytest.approx(layer_level, abs=1e-4)


def test_run_grid_example_math(tiny_model, control_enc, attack_enc):
    layers = enumerate_layers(tiny_model)
    n_heads = tiny_model.config.num_heads
    meta = {"qid": "q1", "docid": "d1", "attack_name": "test"}

    rows = run_grid_example(
        tiny_model, control_enc, attack_enc, layers, TRUE_ID, FALSE_ID, DEVICE, meta, ["zero", "mean"],
    )
    assert rows is not None
    assert len(rows) == len(layers) * n_heads

    for row in rows:
        delta = row["score_attack"] - row["score_control"]
        assert row["fwd_effect"] == pytest.approx((row["score_patched_fwd"] - row["score_control"]) / delta, abs=1e-6)
        assert row["rev_effect"] == pytest.approx((row["score_attack"] - row["score_patched_rev"]) / delta, abs=1e-6)
        assert row["combined_effect"] == pytest.approx(min(row["fwd_effect"], row["rev_effect"]), abs=1e-9)
        # score_ablated_mean == score_patched_rev on the attack base, by construction.
        assert row["score_ablated_mean"] == pytest.approx(row["score_patched_rev"], abs=1e-9)
        assert row["score_drop_mean"] == pytest.approx(row["score_attack"] - row["score_ablated_mean"], abs=1e-9)
        assert row["score_drop_zero"] == pytest.approx(row["score_attack"] - row["score_ablated_zero"], abs=1e-9)


def test_hooks_leave_no_residue(tiny_model, control_enc):
    """After run_grid_example, no forward/pre-hooks should remain on the .o projections."""
    layers = enumerate_layers(tiny_model)
    for layer in layers:
        o_proj = get_o_proj(tiny_model, layer)
        assert len(o_proj._forward_pre_hooks) == 0
        assert len(o_proj._forward_hooks) == 0
