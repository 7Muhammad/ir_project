"""
exp15lib/engine.py
===================
Optimized (batched) edge-message patching engine.

Per example
-----------
1. Two UNPATCHED capture passes (control, attack), each giving the live
   baseline score plus, at every layer that hosts a selected head, P (H,T,T),
   V (H,T,d_kv) and the `.o` input z (T, H*d_kv).
2. Messages m_role[(L, h)](q) = sum_{a in A} P_role[h,q,a] V_role[h,a] for
   every query position q, for role in {control, attack} (edge_messages.py).
   Cheap per-example invariants are asserted here (fail loudly):
     * decomposition  |z_h(q) - sum_j P[h,q,j] V[h,j]| <= decomposition_atol
     * control message ||m_control(q)|| <= control_message_max_norm
3. Patched passes: each batch row is one hypothetical intervention
   (direction, layer, head, set of target query positions). At layer L's `.o`
   input, row r gets z[r, q, h] <- z[r, q, h] - m_receiver(q) + m_donor(q)
   for q in its targets; nothing else is touched. Then the model runs on
   normally (later layers, decoder) and the score is read.

Why precomputed m_receiver is exact (not an approximation)
----------------------------------------------------------
The edit happens at the INPUT of layer L's `.o` projection, i.e. after layer
L's own P and V are computed. Layers < L are untouched. So inside the patched
pass the receiver's layer-L P and V are identical to those of the unpatched
receiver capture pass, and the receiver message computed from the capture
equals the one the patched pass itself would compute. `exp15lib/reference.py`
re-derives P and V live inside the patched pass (independently of
output_attentions / the .v hook / batching / vectorised indexing) and
stage 01 asserts both implementations agree (check N).

Why batching rows is exact
--------------------------
Control and attack encodings have identical length (padded-control
construction), so rows with different receivers, heads, layers and target
sets stack into one batch without padding. Rows never interact inside T5.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

import exp15lib  # noqa: F401
from exp11lib.engine import decoder_pass_scores_batched
from exp11lib.head_hooks import get_attention_module, get_o_proj, head_geometry

from exp15lib.edge_hooks import (
    make_edge_patch_pre_hook,
    make_o_input_capture_pre_hook,
    make_v_capture_hook,
    remove_all,
)
from exp15lib.edge_messages import edge_message, full_head_output, split_heads_values

# direction -> (receiver role, donor role)
DIRECTIONS = {
    "fwd": ("control", "attack"),        # sufficiency: insert attacked message into control
    "rev": ("attack", "control"),        # necessity:  put control message into attacked run
    "noop_control": ("control", "control"),  # sanity only (check C)
    "noop_attack": ("attack", "attack"),     # sanity only (check C)
}


@dataclass
class EdgeRow:
    direction: str            # key of DIRECTIONS
    layer: int
    head: int
    target_q_idx: List[int]   # indices into the example's query position list Q


@dataclass
class ExampleState:
    encs: Dict[str, Dict[str, torch.Tensor]]              # role -> encoding (1, T)
    A: List[int]
    Q: List[int]
    live_scores: Dict[str, float]                          # role -> unpatched live score
    caches: Dict[str, Dict[int, Dict[str, torch.Tensor]]]  # role -> layer -> {P, V, z}
    msgs: Dict[Tuple[str, int, int], torch.Tensor] = field(default_factory=dict)  # (role,L,h) -> (|Q|, d_kv)
    diagnostics: Dict[Tuple[int, int], Dict[str, float]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Capture
# ---------------------------------------------------------------------------

def capture_run(
    model: nn.Module, enc: Dict[str, torch.Tensor], layers: Sequence[int],
    true_id: int, false_id: int,
) -> Tuple[Dict[int, Dict[str, torch.Tensor]], float]:
    """One unpatched encoder pass (+ decoder step) capturing P, V, z at `layers`."""
    n_heads, d_kv, _ = head_geometry(model)
    v_store: Dict[int, torch.Tensor] = {}
    z_store: Dict[int, torch.Tensor] = {}
    handles = []
    try:
        for L in layers:
            attn = get_attention_module(model, L)
            handles.append(attn.v.register_forward_hook(make_v_capture_hook(v_store, L)))
            handles.append(attn.o.register_forward_pre_hook(make_o_input_capture_pre_hook(z_store, L)))
        with torch.no_grad():
            out = model.encoder(
                input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                output_attentions=True,
            )
        score = decoder_pass_scores_batched(
            model, out.last_hidden_state, enc["attention_mask"], true_id, false_id,
        )[0].item()
    finally:
        remove_all(handles)
    if out.attentions is None or len(out.attentions) != len(model.encoder.block):
        raise RuntimeError("encoder did not return per-layer attention probabilities")
    cache = {}
    for L in layers:
        P = out.attentions[L]
        T = enc["input_ids"].shape[1]
        if tuple(P.shape) != (1, n_heads, T, T):
            raise RuntimeError(f"unexpected attention shape {tuple(P.shape)} at layer {L}")
        cache[L] = {
            "P": P[0].detach(),                                        # (H, T, T)
            "V": split_heads_values(v_store[L], n_heads, d_kv),        # (H, T, d_kv)
            "z": z_store[L][0],                                        # (T, H*d_kv)
        }
    return cache, score


def prepare_example(
    model: nn.Module,
    control_enc: Dict[str, torch.Tensor],
    attack_enc: Dict[str, torch.Tensor],
    A: Sequence[int],
    Q: Sequence[int],
    heads: Sequence[Tuple[int, int]],
    true_id: int,
    false_id: int,
    tol: Dict[str, float],
    check_control_zero: bool = True,
) -> ExampleState:
    """
    `check_control_zero=False` is used ONLY by the whole-head equivalence
    sanity check, which sets A = every position (so m_control = z_control is
    legitimately non-zero).
    """
    n_heads, d_kv, _ = head_geometry(model)
    layers = sorted({L for L, _ in heads})
    encs = {"control": control_enc, "attack": attack_enc}
    caches, live = {}, {}
    for role, enc in encs.items():
        caches[role], live[role] = capture_run(model, enc, layers, true_id, false_id)
    st = ExampleState(encs=encs, A=list(A), Q=list(Q), live_scores=live, caches=caches)

    for L, h in heads:
        sl = slice(h * d_kv, (h + 1) * d_kv)
        diag = {}
        for role in ("control", "attack"):
            c = caches[role][L]
            P_h, V_h = c["P"][h], c["V"][h]
            m = edge_message(P_h, V_h, st.A, st.Q)
            st.msgs[(role, L, h)] = m
            dec_err = (c["z"][st.Q, sl] - full_head_output(P_h, V_h, st.Q)).abs().max().item()
            if dec_err > tol["decomposition_atol"]:
                raise RuntimeError(
                    f"decomposition check failed at L{L}H{h} ({role}): max err {dec_err:.3e}"
                )
            diag[f"decomp_err_{role}"] = dec_err
            norms = m.norm(dim=-1)
            diag[f"m_{role}_norm_mean"] = norms.mean().item()
            diag[f"m_{role}_norm_max"] = norms.max().item()
            diag[f"attn_mass_A_{role}_mean"] = c["P"][h][st.Q][:, st.A].sum(-1).mean().item()
        if check_control_zero and diag["m_control_norm_max"] > tol["control_message_max_norm"]:
            raise RuntimeError(
                f"padded-control attack-source message not ~0 at L{L}H{h}: "
                f"max norm {diag['m_control_norm_max']:.3e}"
            )
        st.diagnostics[(L, h)] = diag
    return st


# ---------------------------------------------------------------------------
# Batched patched passes
# ---------------------------------------------------------------------------

def run_edge_rows(
    model: nn.Module,
    st: ExampleState,
    rows: Sequence[EdgeRow],
    true_id: int,
    false_id: int,
    max_rows: int,
    capture_layers: Optional[Sequence[int]] = None,
) -> Tuple[List[float], Dict[int, Dict[str, torch.Tensor]]]:
    """
    Score every row. Returns (scores in row order, captures) where for each
    layer in `capture_layers` captures[L] = {"pre": .o input BEFORE the edit,
    "post": .o input AFTER the edit}, both from the SAME batched pass,
    concatenated over chunks (B_total, T, inner) — used only by tests/sanity
    checks. (Comparing "post" against a batch-1 capture is not bitwise on GPU:
    cuBLAS kernels differ with batch size at the ~1e-6 level.)
    """
    n_heads, d_kv, _ = head_geometry(model)
    device = st.encs["control"]["input_ids"].device
    scores: List[float] = []
    captured: Dict[int, Dict[str, List[torch.Tensor]]] = {L: {"pre": [], "post": []} for L in (capture_layers or [])}

    for start in range(0, len(rows), max_rows):
        chunk = rows[start:start + max_rows]
        ids, masks = [], []
        per_layer: Dict[int, Dict[str, list]] = {}
        for r, row in enumerate(chunk):
            recv, don = DIRECTIONS[row.direction]
            ids.append(st.encs[recv]["input_ids"][0])
            masks.append(st.encs[recv]["attention_mask"][0])
            if not row.target_q_idx:
                raise ValueError("EdgeRow with no targets")
            if len(set(row.target_q_idx)) != len(row.target_q_idx):
                raise ValueError("duplicate target positions in EdgeRow")
            idx = torch.tensor(row.target_q_idx, dtype=torch.long, device=device)
            d = per_layer.setdefault(row.layer, {"row": [], "pos": [], "head": [], "mr": [], "md": []})
            n = len(row.target_q_idx)
            d["row"].append(torch.full((n,), r, dtype=torch.long, device=device))
            d["pos"].append(torch.tensor([st.Q[i] for i in row.target_q_idx], dtype=torch.long, device=device))
            d["head"].append(torch.full((n,), row.head, dtype=torch.long, device=device))
            d["mr"].append(st.msgs[(recv, row.layer, row.head)][idx])
            d["md"].append(st.msgs[(don, row.layer, row.head)][idx])
        input_ids = torch.stack(ids)
        attn_mask = torch.stack(masks)

        store: Dict[int, torch.Tensor] = {}
        store_pre: Dict[int, torch.Tensor] = {}
        handles = []
        try:
            for L in (capture_layers or []):
                # registered BEFORE the patch hook -> sees the unpatched tensor of this same pass
                handles.append(get_o_proj(model, L).register_forward_pre_hook(
                    make_o_input_capture_pre_hook(store_pre, L)))
            for L, d in per_layer.items():
                handles.append(get_o_proj(model, L).register_forward_pre_hook(make_edge_patch_pre_hook(
                    torch.cat(d["row"]), torch.cat(d["pos"]), torch.cat(d["head"]),
                    torch.cat(d["mr"]), torch.cat(d["md"]), n_heads, d_kv,
                )))
            for L in (capture_layers or []):
                # registered AFTER the patch hook -> sees the patched tensor
                handles.append(get_o_proj(model, L).register_forward_pre_hook(
                    make_o_input_capture_pre_hook(store, L)))
            with torch.no_grad():
                out = model.encoder(input_ids=input_ids, attention_mask=attn_mask)
            s = decoder_pass_scores_batched(model, out.last_hidden_state, attn_mask, true_id, false_id)
        finally:
            remove_all(handles)
        scores.extend(s.tolist())
        for L in (capture_layers or []):
            captured[L]["pre"].append(store_pre[L])
            captured[L]["post"].append(store[L])
    return scores, {L: {k: torch.cat(v) for k, v in d.items()} for L, d in captured.items()}


def all_query_rows(heads: Sequence[Tuple[int, int]], n_q: int) -> List[EdgeRow]:
    full = list(range(n_q))
    return [EdgeRow(d, L, h, full) for (L, h) in heads for d in ("fwd", "rev")]


def single_query_rows(heads: Sequence[Tuple[int, int]], n_q: int) -> List[EdgeRow]:
    return [EdgeRow(d, L, h, [i]) for (L, h) in heads for i in range(n_q) for d in ("fwd", "rev")]
