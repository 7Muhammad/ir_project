from __future__ import annotations

import torch

from exp14lib.hooks import make_decoder_shift_pre_hook, make_encoder_masked_shift_pre_hook


def test_decoder_hook_changes_only_requested_head_slice():
    """Test 9: subtraction hook must not touch any other head's slice."""
    n_heads, d_kv = 4, 3
    inner_dim = n_heads * d_kv
    scales = [1.0, 2.0]
    head_idx = 1
    direction = torch.ones(d_kv) * 10.0

    hidden = torch.zeros(len(scales), 1, inner_dim)
    hook = make_decoder_shift_pre_hook(direction, scales, head_idx, d_kv)
    (new_hidden,) = hook(None, (hidden,))

    start, end = head_idx * d_kv, (head_idx + 1) * d_kv
    for row, scale in enumerate(scales):
        # requested head slice shifted by -scale*direction (DEFENSE_SIGN = -1.0)
        expected = -scale * direction
        assert torch.allclose(new_hidden[row, 0, start:end], expected)
        # every other head's slice is untouched (still zero)
        untouched = torch.cat([new_hidden[row, 0, :start], new_hidden[row, 0, end:]])
        assert torch.all(untouched == 0.0)


def test_decoder_hook_scale_zero_is_exact_noop():
    """Test 11 (decoder side): scale 0 must leave the input byte-for-byte identical."""
    n_heads, d_kv = 4, 3
    inner_dim = n_heads * d_kv
    hidden = torch.randn(1, 1, inner_dim)
    hook = make_decoder_shift_pre_hook(torch.ones(d_kv) * 99.0, [0.0], head_idx=2, d_kv=d_kv)
    (new_hidden,) = hook(None, (hidden,))
    assert torch.equal(new_hidden, hidden)


def test_encoder_hook_changes_only_masked_positions():
    """Test 10: encoder hook must not touch positions outside the position mask."""
    n_heads, d_kv, seq_len = 3, 4, 6
    inner_dim = n_heads * d_kv
    head_idx = 0
    direction = torch.ones(d_kv) * 5.0
    scales = [1.0]

    position_mask = torch.zeros(1, seq_len, 1)
    masked_positions = [1, 4]
    for p in masked_positions:
        position_mask[0, p, 0] = 1.0

    hidden = torch.zeros(len(scales), seq_len, inner_dim)
    hook = make_encoder_masked_shift_pre_hook(direction, scales, head_idx, d_kv, position_mask)
    (new_hidden,) = hook(None, (hidden,))

    start, end = head_idx * d_kv, (head_idx + 1) * d_kv
    for pos in range(seq_len):
        slice_val = new_hidden[0, pos, start:end]
        if pos in masked_positions:
            assert torch.allclose(slice_val, -1.0 * direction)
        else:
            assert torch.all(slice_val == 0.0)
    # non-head dimensions are always untouched, at every position
    assert torch.all(new_hidden[:, :, end:] == 0.0)
    assert torch.all(new_hidden[:, :, :start] == 0.0)


def test_encoder_hook_scale_zero_is_exact_noop():
    """Test 11 (encoder side): scale 0 must leave the input byte-for-byte identical, mask notwithstanding."""
    n_heads, d_kv, seq_len = 3, 4, 5
    inner_dim = n_heads * d_kv
    hidden = torch.randn(1, seq_len, inner_dim)
    position_mask = torch.ones(1, seq_len, 1)  # even with everything masked "on"
    hook = make_encoder_masked_shift_pre_hook(torch.ones(d_kv) * 42.0, [0.0], head_idx=1, d_kv=d_kv, position_mask=position_mask)
    (new_hidden,) = hook(None, (hidden,))
    assert torch.equal(new_hidden, hidden)
