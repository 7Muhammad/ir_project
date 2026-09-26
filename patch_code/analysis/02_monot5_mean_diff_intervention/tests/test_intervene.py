"""Tests for exp2lib/intervene.py: the additive per-head shift hook."""

from __future__ import annotations

import torch

from conftest import DEVICE, FALSE_ID, TRUE_ID

from headlib.engine import compute_encoder_states, decoder_pass_scores
from headlib.head_hooks import get_o_proj, head_geometry

from exp2lib.intervene import (
    head_scores_for_shift,
    make_head_shift_pre_hook,
    run_defense_example,
    run_sufficiency_example,
)
from exp2lib.metrics import delta_toward_attack, delta_toward_control


def test_shift_hook_touches_only_target_head_slice():
    n_heads, d_kv = 3, 4
    scales = [0.5, 1.0, 1.5]
    hidden = torch.zeros(len(scales), 1, n_heads * d_kv)
    direction = torch.arange(1.0, d_kv + 1.0)  # [1, 2, 3, 4]

    hook = make_head_shift_pre_hook(direction, scales, sign=-1.0, head_idx=1, d_kv=d_kv)
    (new_hidden,) = hook(None, (hidden,))

    # Head 0 and head 2 slices must be untouched (still zero) in every row.
    assert torch.equal(new_hidden[:, :, 0:d_kv], torch.zeros(len(scales), 1, d_kv))
    assert torch.equal(new_hidden[:, :, 2 * d_kv:3 * d_kv], torch.zeros(len(scales), 1, d_kv))

    # Head 1 slice in row i must equal sign * scales[i] * direction.
    for i, scale in enumerate(scales):
        expected = -1.0 * scale * direction
        assert torch.allclose(new_hidden[i, 0, d_kv:2 * d_kv], expected)


def test_shift_hook_scale_zero_is_a_no_op():
    n_heads, d_kv = 2, 4
    hidden = torch.randn(1, 1, n_heads * d_kv)
    direction = torch.randn(d_kv)
    hook = make_head_shift_pre_hook(direction, [0.0], sign=1.0, head_idx=0, d_kv=d_kv)
    (new_hidden,) = hook(None, (hidden,))
    assert torch.equal(new_hidden, hidden)


def test_head_scores_for_shift_matches_manual_hook_registration(tiny_model, control_enc):
    n_heads, d_kv, _ = head_geometry(tiny_model)
    hidden, mask = compute_encoder_states(tiny_model, control_enc, DEVICE)
    direction = torch.ones(d_kv)
    scales = [0.0, 1.0]

    batched = head_scores_for_shift(
        tiny_model, hidden, mask, "decoder_cross_attn", 0,
        direction, scales, sign=-1.0, head_idx=0, d_kv=d_kv,
        true_id=TRUE_ID, false_id=FALSE_ID,
    )

    # scale=0.0 row must reproduce the unmodified decoder score.
    baseline = decoder_pass_scores(tiny_model, hidden, mask, TRUE_ID, FALSE_ID, 1)[0].item()
    assert abs(batched[0] - baseline) < 1e-5

    # No hooks should remain registered afterwards.
    assert len(get_o_proj(tiny_model, "decoder_cross_attn", 0)._forward_pre_hooks) == 0


def test_metrics_formulas():
    # delta_toward_control: modified score exactly at control -> full recovery.
    assert delta_toward_control(score_attack=1.0, score_control=0.0, score_modified=0.0) == 1.0
    # no change at all -> zero delta.
    assert delta_toward_control(score_attack=1.0, score_control=0.0, score_modified=1.0) == 0.0
    # overcorrecting FURTHER from control than the attack score was is penalised.
    assert delta_toward_control(score_attack=1.0, score_control=0.0, score_modified=-2.0) < 0

    assert delta_toward_attack(score_clean=0.0, score_attack=1.0, score_modified=1.0) == 1.0
    assert delta_toward_attack(score_clean=0.0, score_attack=1.0, score_modified=0.0) == 0.0


def test_run_defense_and_sufficiency_rows_shape(tiny_model, control_enc, attack_enc, clean_enc):
    n_heads, d_kv, _ = head_geometry(tiny_model)
    flagged_heads = [{"layer": 0, "component": "decoder_cross_attn", "head_idx": 0}]
    key = ("per_head", "decoder_cross_attn", 0, 0)
    directions = {key: torch.ones(d_kv)}
    direction_norms = {key: float(torch.linalg.norm(torch.ones(d_kv)).item())}
    scales = [0.5, 1.0, 1.5]
    meta = {
        "qid": "q1", "docid": "d1", "attack_name": "relevant_start_5",
        "score_clean": 0.1, "score_control": 0.2, "score_attack": 0.9,
    }

    defense_rows = run_defense_example(
        tiny_model, attack_enc, flagged_heads, directions, direction_norms,
        scales, TRUE_ID, FALSE_ID, DEVICE, meta,
    )
    assert len(defense_rows) == len(scales)
    for row in defense_rows:
        assert row["intervention_type"] == "subtract"
        assert row["head_idx"] == 0

    sufficiency_rows = run_sufficiency_example(
        tiny_model, clean_enc, flagged_heads, directions, direction_norms,
        scales, TRUE_ID, FALSE_ID, DEVICE, meta,
    )
    assert len(sufficiency_rows) == len(scales)
    for row in sufficiency_rows:
        assert row["intervention_type"] == "add"
