"""
Correctness tests for the per-head patching engine.

The important properties:
  1. Encoder-output reuse is exact (same score as a full forward).
  2. Batched decoder rows are independent and equal the single-row score.
  3. The batched per-row head hook equals patching one head at a time.
  4. Replacing ALL head slices at the .o input reproduces Experiment 1's
     layer-level contribution patch (consistency across experiments).
  5. Zeroing all head slices equals a zero-contribution layer patch
     (W_o has no bias).
  6. run_grid_example's effect math is internally consistent, and
     score_ablated_mean == score_patched_rev on the attack base.
"""

from __future__ import annotations

import pytest
import torch

from conftest import DEVICE, FALSE_ID, TRUE_ID

from src.activation_hooks import (
    get_module_for_component,
    make_cache_hook,
    make_patch_hook,
)
from src.model_utils import score_from_encoding

from headlib.engine import (
    cache_head_inputs,
    compute_encoder_states,
    decoder_pass_scores,
    head_scores_for_slot,
    run_clean_example,
    run_grid_example,
)
from headlib.head_hooks import (
    block_diag_head_mask,
    enumerate_slots,
    get_o_proj,
    head_geometry,
    make_full_replace_pre_hook,
    make_per_row_head_pre_hook,
    slot_key,
)

# The Experiment-1 component name for each head component (same module).
EXP1_COMPONENT = {
    "decoder_self_attn": "decoder_self_attn",
    "decoder_cross_attn": "decoder_cross_attn",
}


def test_encoder_reuse_matches_full_forward(tiny_model, control_enc):
    full = score_from_encoding(tiny_model, control_enc, TRUE_ID, FALSE_ID, DEVICE)
    hidden, mask = compute_encoder_states(tiny_model, control_enc, DEVICE)
    reused = decoder_pass_scores(tiny_model, hidden, mask, TRUE_ID, FALSE_ID, 1)[0].item()
    assert reused == pytest.approx(full, abs=1e-5)


def test_batched_rows_equal_single(tiny_model, attack_enc):
    hidden, mask = compute_encoder_states(tiny_model, attack_enc, DEVICE)
    single = decoder_pass_scores(tiny_model, hidden, mask, TRUE_ID, FALSE_ID, 1)
    batched = decoder_pass_scores(tiny_model, hidden, mask, TRUE_ID, FALSE_ID, 5)
    assert torch.allclose(batched, single.expand(5), atol=1e-5)


def test_cache_shapes_and_base_score(tiny_model, control_enc):
    slots = enumerate_slots(tiny_model)
    n_heads, d_kv, inner = head_geometry(tiny_model)
    hidden, mask = compute_encoder_states(tiny_model, control_enc, DEVICE)
    cache, score = cache_head_inputs(tiny_model, hidden, mask, slots, TRUE_ID, FALSE_ID)
    assert set(cache) == {slot_key(c, l) for c, l in slots}
    for t in cache.values():
        assert t.shape == (1, 1, inner)
    assert score == pytest.approx(
        score_from_encoding(tiny_model, control_enc, TRUE_ID, FALSE_ID, DEVICE), abs=1e-5
    )


def test_batched_head_hook_equals_one_head_at_a_time(tiny_model, control_enc, attack_enc):
    """Row h of the batched pass must equal a single-row pass patching only head h."""
    slots = enumerate_slots(tiny_model)
    n_heads, d_kv, _ = head_geometry(tiny_model)
    head_mask = block_diag_head_mask(n_heads, d_kv, DEVICE)

    ctrl_hidden, ctrl_mask = compute_encoder_states(tiny_model, control_enc, DEVICE)
    atk_hidden, atk_mask = compute_encoder_states(tiny_model, attack_enc, DEVICE)
    atk_cache, _ = cache_head_inputs(tiny_model, atk_hidden, atk_mask, slots, TRUE_ID, FALSE_ID)

    comp, layer = "decoder_cross_attn", 1
    repl = atk_cache[slot_key(comp, layer)]
    batched = head_scores_for_slot(
        tiny_model, ctrl_hidden, ctrl_mask, comp, layer, repl, head_mask, TRUE_ID, FALSE_ID
    )

    o_proj = get_o_proj(tiny_model, comp, layer)
    for h in range(n_heads):
        def one_head_hook(module, args, h=h):
            hid = args[0].clone()
            hid[:, :, h * d_kv:(h + 1) * d_kv] = repl[:, :, h * d_kv:(h + 1) * d_kv]
            return (hid,) + args[1:]

        handle = o_proj.register_forward_pre_hook(one_head_hook)
        try:
            single = decoder_pass_scores(
                tiny_model, ctrl_hidden, ctrl_mask, TRUE_ID, FALSE_ID, 1
            )[0].item()
        finally:
            handle.remove()
        assert batched[h] == pytest.approx(single, abs=1e-5), f"head {h} mismatch"

    # Patching must actually change something for at least one head.
    base = decoder_pass_scores(tiny_model, ctrl_hidden, ctrl_mask, TRUE_ID, FALSE_ID, 1)[0].item()
    assert any(abs(s - base) > 1e-7 for s in batched)


@pytest.mark.parametrize("comp,layer", [
    ("decoder_self_attn", 0),
    ("decoder_cross_attn", 0),
    ("decoder_self_attn", 1),
    ("decoder_cross_attn", 1),
])
def test_all_heads_patch_equals_exp1_layer_patch(tiny_model, control_enc, attack_enc, comp, layer):
    """
    Replacing every head slice at the .o input with the attack run's slices
    must equal Experiment 1's layer-level contribution patch of the same
    sub-layer (make_cache_hook / make_patch_hook from src/activation_hooks.py).
    """
    slots = enumerate_slots(tiny_model)
    atk_hidden, atk_mask = compute_encoder_states(tiny_model, attack_enc, DEVICE)
    atk_cache, _ = cache_head_inputs(tiny_model, atk_hidden, atk_mask, slots, TRUE_ID, FALSE_ID)

    # Head-level: replace the full .o input on the control base run.
    ctrl_hidden, ctrl_mask = compute_encoder_states(tiny_model, control_enc, DEVICE)
    handle = get_o_proj(tiny_model, comp, layer).register_forward_pre_hook(
        make_full_replace_pre_hook(atk_cache[slot_key(comp, layer)])
    )
    try:
        head_level = decoder_pass_scores(
            tiny_model, ctrl_hidden, ctrl_mask, TRUE_ID, FALSE_ID, 1
        )[0].item()
    finally:
        handle.remove()

    # Experiment-1 layer level: cache the attack contribution of the whole
    # sub-layer, then patch it into the control run.
    module = get_module_for_component(tiny_model, EXP1_COMPONENT[comp], layer)
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
    """W_o has no bias, so zeroing every head slice == zeroing the sub-layer's
    residual contribution (Experiment-1 patch with a zero tensor)."""
    comp, layer = "decoder_cross_attn", 1
    n_heads, d_kv, inner = head_geometry(tiny_model)
    atk_hidden, atk_mask = compute_encoder_states(tiny_model, attack_enc, DEVICE)

    handle = get_o_proj(tiny_model, comp, layer).register_forward_pre_hook(
        make_full_replace_pre_hook(torch.zeros(1, 1, inner))
    )
    try:
        head_level = decoder_pass_scores(
            tiny_model, atk_hidden, atk_mask, TRUE_ID, FALSE_ID, 1
        )[0].item()
    finally:
        handle.remove()

    module = get_module_for_component(tiny_model, comp, layer)
    d_model = tiny_model.config.d_model
    handle = module.register_forward_hook(make_patch_hook(torch.zeros(1, 1, d_model)))
    try:
        layer_level = score_from_encoding(tiny_model, attack_enc, TRUE_ID, FALSE_ID, DEVICE)
    finally:
        handle.remove()

    assert head_level == pytest.approx(layer_level, abs=1e-4)


def test_per_row_hook_zero_mode(tiny_model, attack_enc):
    """Zero mode of the batched hook: row h must zero exactly head h's slice."""
    n_heads, d_kv, inner = head_geometry(tiny_model)
    head_mask = block_diag_head_mask(n_heads, d_kv, DEVICE)
    hook = make_per_row_head_pre_hook(None, head_mask)
    hidden = torch.ones(n_heads, 1, inner)
    (out,) = hook(None, (hidden,))
    for h in range(n_heads):
        sl = slice(h * d_kv, (h + 1) * d_kv)
        assert torch.all(out[h, :, sl] == 0)
        outside = out[h].clone()
        outside[:, sl] = 1.0
        assert torch.all(outside == 1.0)


def test_run_grid_example_math(tiny_model, control_enc, attack_enc):
    slots = enumerate_slots(tiny_model)
    n_heads, _, _ = head_geometry(tiny_model)
    meta = {"qid": "q1", "docid": "d1", "attack_name": "unit_test",
            "score_clean": 0.0, "score_control": None, "score_attack": None}
    rows = run_grid_example(
        tiny_model, control_enc, attack_enc, slots,
        TRUE_ID, FALSE_ID, DEVICE, meta, methods=["zero", "mean"],
    )
    assert rows is not None
    assert len(rows) == len(slots) * n_heads

    for r in rows:
        delta = r["score_attack"] - r["score_control"]
        assert abs(delta) >= 1e-4
        assert r["fwd_effect"] == pytest.approx(
            (r["score_patched_fwd"] - r["score_control"]) / delta, abs=1e-9)
        assert r["rev_effect"] == pytest.approx(
            (r["score_attack"] - r["score_patched_rev"]) / delta, abs=1e-9)
        assert r["combined_effect"] == pytest.approx(
            min(r["fwd_effect"], r["rev_effect"]), abs=1e-12)
        # mean ablation on the attack base IS the reverse patch
        assert r["score_ablated_mean"] == r["score_patched_rev"]
        assert r["score_drop_zero"] == pytest.approx(
            r["score_attack"] - r["score_ablated_zero"], abs=1e-9)
        assert r["score_drop_mean"] == pytest.approx(
            r["score_attack"] - r["score_ablated_mean"], abs=1e-9)

    # base scores must match the plain scoring function
    ctrl = score_from_encoding(tiny_model, control_enc, TRUE_ID, FALSE_ID, DEVICE)
    atk = score_from_encoding(tiny_model, attack_enc, TRUE_ID, FALSE_ID, DEVICE)
    assert rows[0]["score_control"] == pytest.approx(ctrl, abs=1e-5)
    assert rows[0]["score_attack"] == pytest.approx(atk, abs=1e-5)


def test_run_clean_example(tiny_model, clean_enc, control_enc):
    """Clean side-effect run: different-length clean input, drops vs clean score."""
    slots = enumerate_slots(tiny_model)
    n_heads, _, _ = head_geometry(tiny_model)
    meta = {"qid": "q1", "docid": "d1", "attack_name": "unit_test",
            "score_attack": 1.23, "score_control": None}
    rows = run_clean_example(
        tiny_model, clean_enc, control_enc, slots,
        TRUE_ID, FALSE_ID, DEVICE, meta, methods=["zero", "mean"],
    )
    assert len(rows) == len(slots) * n_heads

    clean = score_from_encoding(tiny_model, clean_enc, TRUE_ID, FALSE_ID, DEVICE)
    for r in rows:
        assert r["score_clean"] == pytest.approx(clean, abs=1e-5)
        assert r["score_attack"] == 1.23
        assert r["score_drop_zero"] == pytest.approx(
            r["score_clean"] - r["score_ablated_zero"], abs=1e-9)
        assert r["score_drop_mean"] == pytest.approx(
            r["score_clean"] - r["score_ablated_mean"], abs=1e-9)


def test_hooks_leave_no_residue(tiny_model, attack_enc):
    """After any engine call, an unhooked pass must reproduce the base score."""
    slots = enumerate_slots(tiny_model)
    n_heads, d_kv, _ = head_geometry(tiny_model)
    head_mask = block_diag_head_mask(n_heads, d_kv, DEVICE)
    base = score_from_encoding(tiny_model, attack_enc, TRUE_ID, FALSE_ID, DEVICE)

    hidden, mask = compute_encoder_states(tiny_model, attack_enc, DEVICE)
    cache_head_inputs(tiny_model, hidden, mask, slots, TRUE_ID, FALSE_ID)
    head_scores_for_slot(tiny_model, hidden, mask, "decoder_self_attn", 0,
                         None, head_mask, TRUE_ID, FALSE_ID)

    after = score_from_encoding(tiny_model, attack_enc, TRUE_ID, FALSE_ID, DEVICE)
    assert after == pytest.approx(base, abs=1e-6)
