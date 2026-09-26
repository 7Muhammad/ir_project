from __future__ import annotations

import pandas as pd

from exp14lib.scale_selection import select_scales


def _row(side, layer, head_idx, condition, scale, recovery):
    return {"side": side, "layer": layer, "head_idx": head_idx, "condition": condition, "scale": scale, "recovery": recovery}


def test_select_scales_picks_highest_mean_recovery():
    rows = [
        _row("decoder", 11, 3, "decoder", 0.5, recovery=0.1),
        _row("decoder", 11, 3, "decoder", 0.5, recovery=0.3),
        _row("decoder", 11, 3, "decoder", 1.0, recovery=0.4),
        _row("decoder", 11, 3, "decoder", 1.0, recovery=0.6),
        _row("decoder", 11, 3, "decoder", 1.5, recovery=0.05),
        _row("decoder", 11, 3, "decoder", 1.5, recovery=0.05),
    ]
    df = pd.DataFrame(rows)
    selected = select_scales(df)
    key = "decoder:11:3:decoder"
    assert key in selected
    assert selected[key]["selected_scale"] == 1.0  # mean 0.5, beats scale=0.5 (mean 0.2) and scale=1.5 (mean 0.05)
    assert selected[key]["mean_recovery_at_selected_scale"] == 0.5


def test_select_scales_tie_break_prefers_smaller_scale():
    rows = [
        _row("encoder", 10, 0, "document", 1.0, recovery=0.5),
        _row("encoder", 10, 0, "document", 2.0, recovery=0.5),  # exact tie with scale=1.0
        _row("encoder", 10, 0, "document", 3.0, recovery=0.2),
    ]
    df = pd.DataFrame(rows)
    selected = select_scales(df)
    key = "encoder:10:0:document"
    assert selected[key]["selected_scale"] == 1.0  # smaller of the tied scales


def test_select_scales_handles_multiple_independent_candidates():
    rows = [
        _row("decoder", 11, 3, "decoder", 1.0, recovery=0.9),
        _row("encoder", 10, 0, "query", 1.0, recovery=0.1),
        _row("encoder", 10, 0, "query", 2.0, recovery=0.4),
    ]
    df = pd.DataFrame(rows)
    selected = select_scales(df)
    assert len(selected) == 2
    assert selected["decoder:11:3:decoder"]["selected_scale"] == 1.0
    assert selected["encoder:10:0:query"]["selected_scale"] == 2.0
