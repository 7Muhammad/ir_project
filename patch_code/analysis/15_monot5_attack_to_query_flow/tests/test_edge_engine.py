"""
Checks A-H, M, N on a tiny random T5 (exact same code paths as the real run).

  A decomposition           B control message ~ 0     C no-op donor
  D target isolation        E head isolation          F all-query mode
  G forward formula         H reverse formula
  M whole-head reference    N optimized == reference  + batching invariance
"""

from __future__ import annotations

import pytest
import torch

from conftest import A_POS, FALSE_ID, Q_POS, T, TRUE_ID

from exp11lib.head_hooks import head_geometry
from exp15lib.edge_messages import edge_message, full_head_output
from exp15lib.engine import (
    DIRECTIONS, EdgeRow, all_query_rows, capture_run, prepare_example, run_edge_rows, single_query_rows,
)
from exp15lib.reference import reference_patched_score
from exp15lib.whole_head import whole_head_scores

HEADS = [(1, 2), (2, 0)]


@pytest.fixture(scope="module")
def state(tiny_model, encs, tol):
    return prepare_example(tiny_model, encs["control"], encs["attack"], A_POS, Q_POS, HEADS, TRUE_ID, FALSE_ID, tol)


def test_A_decomposition_and_message_shape(tiny_model, encs):
    cache, _ = capture_run(tiny_model, encs["attack"], [1], TRUE_ID, FALSE_ID)
    n_heads, d_kv, inner = head_geometry(tiny_model)
    P, V, z = cache[1]["P"], cache[1]["V"], cache[1]["z"]
    assert P.shape == (n_heads, T, T) and V.shape == (n_heads, T, d_kv) and z.shape == (T, inner)
    for h in range(n_heads):
        full = full_head_output(P[h], V[h], range(T))
        assert torch.allclose(full, z[:, h * d_kv:(h + 1) * d_kv], atol=1e-6)
        q = Q_POS[1]
        m = edge_message(P[h], V[h], A_POS, [q])
        manual = sum(P[h, q, a] * V[h, a] for a in A_POS)
        assert m.shape == (1, d_kv)
        assert torch.allclose(m[0], manual, atol=1e-7)


def test_B_control_message_is_zero_attack_is_not(state):
    for L, h in HEADS:
        assert state.msgs[("control", L, h)].norm(dim=-1).max() <= 1e-6
        assert state.msgs[("attack", L, h)].norm(dim=-1).max() > 1e-4
        assert state.caches["control"][L]["P"][h][:, A_POS].abs().max() == 0.0


def test_C_noop_donor_leaves_score_unchanged(tiny_model, state):
    rows = [EdgeRow(d, L, h, list(range(len(Q_POS)))) for L, h in HEADS for d in ("noop_control", "noop_attack")]
    scores, _ = run_edge_rows(tiny_model, state, rows, TRUE_ID, FALSE_ID, 8)
    for r, s in zip(rows, scores):
        assert s == pytest.approx(state.live_scores[DIRECTIONS[r.direction][0]], abs=1e-5)


def _captured(tiny_model, state, row):
    _, cap = run_edge_rows(tiny_model, state, [row], TRUE_ID, FALSE_ID, 8, capture_layers=[row.layer])
    return cap[row.layer]["pre"][0], cap[row.layer]["post"][0]


@pytest.mark.parametrize("direction", ["fwd", "rev"])
@pytest.mark.parametrize("targets", [[0], [2], [0, 1, 2, 3]])   # single token (D) and all-query (F)
def test_DEFGH_isolation_and_formula(tiny_model, state, direction, targets):
    _, d_kv, _ = head_geometry(tiny_model)
    L, h = HEADS[0]
    row = EdgeRow(direction, L, h, targets)
    pre, got = _captured(tiny_model, state, row)
    recv, don = DIRECTIONS[direction]
    assert (recv, don) == {"fwd": ("control", "attack"), "rev": ("attack", "control")}[direction]
    # the same-pass unpatched input is the receiver's own run (identical to its capture here on CPU)
    assert torch.allclose(pre, state.caches[recv][L]["z"], atol=1e-6)
    z = pre
    tpos = [Q_POS[i] for i in targets]
    sl = slice(h * d_kv, (h + 1) * d_kv)
    changed = torch.zeros_like(z, dtype=torch.bool)
    changed[tpos, sl] = True
    # D/E/F: every other position (query, doc, template) and every other head: bit-identical
    assert torch.equal(got[~changed], z[~changed])
    # G/H: exact formula at the targets
    idx = torch.tensor(targets)
    expect = z[tpos][:, sl] - state.msgs[(recv, L, h)][idx] + state.msgs[(don, L, h)][idx]
    assert torch.allclose(got[tpos][:, sl], expect, atol=1e-7)
    # the edit is real (attack message non-zero)
    assert (got[tpos][:, sl] - z[tpos][:, sl]).abs().max() > 1e-5


def test_F_all_query_is_explicit_not_sum(tiny_model, state):
    L, h = HEADS[0]
    rows = all_query_rows([(L, h)], len(Q_POS))
    assert all(r.target_q_idx == list(range(len(Q_POS))) for r in rows)
    srows = single_query_rows([(L, h)], len(Q_POS))
    assert all(len(r.target_q_idx) == 1 for r in srows) and len(srows) == 2 * len(Q_POS)


def test_N_optimized_matches_reference(tiny_model, state, encs):
    rows = all_query_rows(HEADS, len(Q_POS)) + single_query_rows(HEADS, len(Q_POS))
    scores, _ = run_edge_rows(tiny_model, state, rows, TRUE_ID, FALSE_ID, 5)
    for r, s in zip(rows, scores):
        ref = reference_patched_score(tiny_model, encs, r.direction, r.layer, r.head, A_POS,
                                      [Q_POS[i] for i in r.target_q_idx], TRUE_ID, FALSE_ID)
        assert s == pytest.approx(ref, abs=1e-5)


def test_batching_invariance(tiny_model, state):
    rows = single_query_rows(HEADS, len(Q_POS))
    a, _ = run_edge_rows(tiny_model, state, rows, TRUE_ID, FALSE_ID, 1)
    b, _ = run_edge_rows(tiny_model, state, rows, TRUE_ID, FALSE_ID, 64)
    assert a == pytest.approx(b, abs=1e-5)


def test_M_whole_head_reference_equals_edge_with_all_sources(tiny_model, encs, tol):
    """Edge patching with A = Q = all positions is exactly Exp 11's whole-head patch."""
    st = prepare_example(tiny_model, encs["control"], encs["attack"], list(range(T)), list(range(T)),
                         HEADS, TRUE_ID, FALSE_ID, tol, check_control_zero=False)
    rows = all_query_rows(HEADS, T)
    scores, _ = run_edge_rows(tiny_model, st, rows, TRUE_ID, FALSE_ID, 8)
    wh, live_c, live_a = whole_head_scores(tiny_model, encs["control"], encs["attack"], HEADS,
                                           torch.device("cpu"), TRUE_ID, FALSE_ID)
    for r, s in zip(rows, scores):
        assert s == pytest.approx(wh[(r.layer, r.head)][0 if r.direction == "fwd" else 1], abs=1e-5)
    assert live_c == pytest.approx(st.live_scores["control"], abs=1e-6)
    assert live_a == pytest.approx(st.live_scores["attack"], abs=1e-6)


def test_whole_head_subset_mask_matches_full_exp11_layer(tiny_model, encs):
    """Passing only canonical heads' mask rows == picking those rows from Exp 11's full 12-row pass."""
    from exp11lib.engine import cache_head_inputs, head_scores_for_layer
    from exp11lib.head_hooks import block_diag_head_mask, slot_key
    n_heads, d_kv, _ = head_geometry(tiny_model)
    dev = torch.device("cpu")
    wh, _, _ = whole_head_scores(tiny_model, encs["control"], encs["attack"], HEADS, dev, TRUE_ID, FALSE_ID)
    atk_cache, _ = cache_head_inputs(tiny_model, encs["attack"], dev, [1, 2], TRUE_ID, FALSE_ID)
    for L, h in HEADS:
        full = head_scores_for_layer(tiny_model, encs["control"], dev, L, atk_cache[slot_key(L)],
                                     block_diag_head_mask(n_heads, d_kv, dev), TRUE_ID, FALSE_ID)
        assert wh[(L, h)][0] == pytest.approx(full[h], abs=1e-5)
