"""
exp13lib/path_engine.py
=========================
True encoder-head -> decoder-cross-attention-head PATH patching (Experiment
13 / "2A"). Reuses Experiment 11's encoder per-head hook machinery and
Experiment 3's decoder per-head hook machinery; the only genuinely new
mechanism here is the "selective receiver" hook that lets exactly ONE
decoder cross-attention LAYER read a substituted (hybrid) encoder
representation while every other layer keeps reading the baseline one
(see ``cross_attn_kv_swap_pre_hook`` below).

Why this is TRUE path patching, not two independent patches
----------------------------------------------------------------
"Patch encoder head A, observe the score" (Experiment 11 alone) tells you
A matters. "Patch encoder head A AND independently patch decoder head B"
tells you both matter, but not that A's effect reaches the score
*through* B specifically -- B might read the ordinary control/attack
encoder state and still show an effect for reasons unrelated to A.

Path patching isolates the single route A -> encoder -> B -> score:
  1. Build a "hybrid" encoder representation: start from the BASE run's
     own encoder computation, but splice in sender head A's activation
     from the OTHER run at layer A, then let the rest of the encoder
     process that splice normally. This is exactly Experiment 11's
     per-head patch, except we keep the *propagated final hidden state*
     instead of just reading off the score.
  2. Feed that hybrid representation to receiver head B ONLY. Every
     other decoder cross-attention head (same layer or any other layer)
     keeps reading the ordinary baseline encoder representation.
  3. Let the decoder finish computing normally from B onward (later
     layers, final layernorm, lm_head) so downstream propagation is
     real, not truncated.

Step 2 in two sub-steps (both exact, not an approximation)
------------------------------------------------------------
T5 multi-head attention has NO cross-head interaction until the final
concatenation + output projection (`.o`): each head's own output depends
only on its own per-head Q/K/V, never on any other head's K/V source.
So "receiver head B alone reads the hybrid K/V, every other head at its
layer reads baseline K/V" is *exactly* equal to "the WHOLE layer reads
hybrid K/V, then take only head B's slice of the result" -- the other
heads' baseline-K/V outputs are simply discarded, not altered.

That means, per (sender A, receiver LAYER L, direction), we can:
  Step B: run ONE decoder pass whose encoder input is baseline
          everywhere EXCEPT layer L's cross-attention module, whose
          `key_value_states` kwarg is swapped (via a forward pre-hook on
          the T5LayerCrossAttention module -- NOT the `.o` projection)
          to the hybrid encoder representation. Cache the resulting
          `.o`-input at layer L (all heads at once).
  Step D: run the EXISTING headlib per-head batched-row `.o`-input
          patch (Experiment 3's own `head_scores_for_slot`), using Step
          B's cached tensor as the "replacement" and the standard
          block-diagonal per-head mask -- this reproduces, for every
          head index h at layer L in one batched pass, "baseline
          everywhere except head h's slice <- Step-B's head-h slice",
          then lets the decoder continue normally past layer L.
Step B is cached per (sender, receiver LAYER, direction) and reused for
every receiver head at that layer (up to several of the 31 receivers
share a layer), and step D is naturally batched over all 12 heads of
that layer in a single decoder pass -- we simply read off whichever
head indices are in our receiver set. The encoder-side hybrid itself
(Step 1) is built once per sender per example and reused across every
receiver layer and direction pairing that needs it.
"""

from __future__ import annotations

import pathlib
import sys
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers.modeling_outputs import BaseModelOutput

EXP_DIR = pathlib.Path(__file__).parent.parent
EXP1_DIR = EXP_DIR.parent / "01_monot5_layer_patching"
EXP3_DIR = EXP_DIR.parent / "03_monot5_head_patching_ablation"
EXP11_DIR = EXP_DIR.parent / "11_monot5_encoder_head_patching"
for _p in (EXP1_DIR, EXP3_DIR, EXP11_DIR, EXP_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from src.patching import SKIP_EPSILON  # noqa: E402

import exp11lib.head_hooks as enc_hooks  # noqa: E402
import headlib.head_hooks as dec_hooks  # noqa: E402
from headlib.engine import decoder_pass_scores, head_scores_for_slot  # noqa: E402

from exp13lib.head_lists import ReceiverHead, SenderHead  # noqa: E402

__all__ = [
    "SKIP_EPSILON",
    "cache_encoder_and_score",
    "build_encoder_hybrid",
    "cross_attn_kv_swap_pre_hook",
    "compute_receiver_layer_step_b_cache",
    "run_example_paths",
    "group_receivers_by_layer",
]


# ---------------------------------------------------------------------------
# Step 1 helpers: encoder-side hybrid construction (reuses exp11lib hooks)
# ---------------------------------------------------------------------------

def cache_encoder_and_score(
    model: nn.Module,
    enc: Dict[str, torch.Tensor],
    device: torch.device,
    layers: List[int],
    true_id: int,
    false_id: int,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, float]:
    """
    Run ONE unpatched encoder forward pass, caching the `.o`-input
    (concatenated per-head outputs) at every requested encoder layer AND
    returning the final encoder hidden state (needed as the decoder-side
    "baseline" for Step B/D below) plus the base monoT5 score.

    Adapted from exp11lib.engine.cache_head_inputs, which caches the
    per-layer `.o` inputs and the base score but does not expose the
    final hidden state -- we need that too, to avoid a second redundant
    encoder forward pass for the same (unpatched) input.
    """
    cache: Dict[str, torch.Tensor] = {}
    handles = []
    enc_dev = {k: v.to(device) for k, v in enc.items()}
    try:
        for layer in layers:
            handles.append(
                enc_hooks.get_o_proj(model, layer).register_forward_pre_hook(
                    enc_hooks.make_cache_pre_hook(cache, enc_hooks.slot_key(layer))
                )
            )
        with torch.no_grad():
            enc_out = model.encoder(
                input_ids=enc_dev["input_ids"], attention_mask=enc_dev["attention_mask"],
            )
        score = decoder_pass_scores(
            model, enc_out.last_hidden_state, enc_dev["attention_mask"], true_id, false_id, batch_rows=1,
        )[0].item()
    finally:
        for h in handles:
            h.remove()
    return cache, enc_out.last_hidden_state, enc_dev["attention_mask"], score


def build_encoder_hybrid(
    model: nn.Module,
    base_enc: Dict[str, torch.Tensor],
    source_layer_o_cache: torch.Tensor,
    sender_layer: int,
    single_head_mask: torch.Tensor,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build ``encoder_hybrid_forward(A)`` / ``encoder_hybrid_reverse(A)``.

    Starts a fresh encoder forward pass on `base_enc` (the padded-control
    tokens for the forward direction, the attacked tokens for reverse),
    with a pre-hook at `sender_layer`'s `.o` projection that replaces
    ONLY sender head A's slice (selected by `single_head_mask`, shape
    (1, 1, inner_dim)) with the corresponding slice of
    `source_layer_o_cache` (the OTHER run's cached `.o`-input at that
    same layer -- attack's for forward, control's for reverse). The
    patched activation then propagates through the remaining encoder
    layers normally (this hook only touches sender_layer; no other
    encoder layer's forward call is intercepted).

    Returns (last_hidden_state, attention_mask) of the hybrid encoder
    representation, with the SAME shape/mask as `base_enc` (same input
    tokens, only internal head activations differ) -- so it can be fed
    directly wherever `base_enc`'s own encoder output would be used.
    """
    base_dev = {k: v.to(device) for k, v in base_enc.items()}
    hook = enc_hooks.make_per_row_head_pre_hook(source_layer_o_cache, single_head_mask)
    handle = enc_hooks.get_o_proj(model, sender_layer).register_forward_pre_hook(hook)
    try:
        with torch.no_grad():
            out = model.encoder(input_ids=base_dev["input_ids"], attention_mask=base_dev["attention_mask"])
    finally:
        handle.remove()
    return out.last_hidden_state, base_dev["attention_mask"]


# ---------------------------------------------------------------------------
# Step 2/3 helpers: selective receiver isolation (new mechanism)
# ---------------------------------------------------------------------------

def cross_attn_kv_swap_pre_hook(hybrid_encoder_hidden: torch.Tensor):
    """
    Forward PRE-hook (with_kwargs=True) for ONE decoder layer's
    T5LayerCrossAttention module (``model.decoder.block[L].layer[1]``).

    T5Block calls this module as
        self.layer[1](hidden_states, key_value_states=encoder_hidden_states, ...)
    i.e. `key_value_states` arrives as a KEYWORD argument (see
    transformers modeling_t5.py T5Block.forward / T5LayerCrossAttention.forward).
    Replacing it here substitutes the encoder representation that layer
    L's cross-attention (ALL of its heads) computes K/V from, while every
    OTHER decoder layer -- hooked or not -- keeps using whatever
    `encoder_hidden_states` was passed to the outer `model(...)` call.
    Because this hook is registered on ONE specific layer's module
    instance, no other layer's cross-attention is touched.
    """

    def hook(module: nn.Module, args: tuple, kwargs: dict) -> Tuple[tuple, dict]:
        orig = kwargs.get("key_value_states")
        new_kwargs = dict(kwargs)
        new_kwargs["key_value_states"] = hybrid_encoder_hidden.to(dtype=orig.dtype, device=orig.device)
        return args, new_kwargs

    return hook


def compute_receiver_layer_step_b_cache(
    model: nn.Module,
    base_hidden: torch.Tensor,
    base_mask: torch.Tensor,
    hybrid_hidden: torch.Tensor,
    receiver_layer: int,
    true_id: int,
    false_id: int,
    device: torch.device,
) -> torch.Tensor:
    """
    "Step B": one decoder forward pass over `base_hidden`/`base_mask`
    (the plain control or attack encoder representation, batch=1) with
    layer `receiver_layer`'s cross-attention K/V swapped to
    `hybrid_hidden`. Every OTHER layer -- both earlier layers (which
    determine the decoder hidden state feeding INTO receiver_layer's
    query projection) and later layers -- reads `base_hidden` exactly as
    in an ordinary unpatched pass; only receiver_layer's `key_value_states`
    kwarg is substituted.

    Returns the resulting `.o`-input at receiver_layer (ALL heads,
    shape (1, 1, inner_dim)); Step D (headlib.engine.head_scores_for_slot)
    then extracts individual receiver heads' slices from it.
    """
    cache: Dict[str, torch.Tensor] = {}
    handles = []
    try:
        handles.append(
            dec_hooks.get_o_proj(model, "decoder_cross_attn", receiver_layer).register_forward_pre_hook(
                dec_hooks.make_cache_pre_hook(cache, dec_hooks.slot_key("decoder_cross_attn", receiver_layer))
            )
        )
        cross_attn_module = model.decoder.block[receiver_layer].layer[1]
        handles.append(
            cross_attn_module.register_forward_pre_hook(
                cross_attn_kv_swap_pre_hook(hybrid_hidden), with_kwargs=True
            )
        )
        decoder_pass_scores(model, base_hidden, base_mask, true_id, false_id, batch_rows=1)
    finally:
        for h in handles:
            h.remove()
    return cache[dec_hooks.slot_key("decoder_cross_attn", receiver_layer)]


# ---------------------------------------------------------------------------
# Per-example driver
# ---------------------------------------------------------------------------

def group_receivers_by_layer(receivers: List[ReceiverHead]) -> Dict[int, List[ReceiverHead]]:
    grouped: Dict[int, List[ReceiverHead]] = {}
    for r in receivers:
        grouped.setdefault(r.layer, []).append(r)
    return grouped


def run_example_paths(
    model: nn.Module,
    control_enc: Dict[str, torch.Tensor],
    attack_enc: Dict[str, torch.Tensor],
    senders: List[SenderHead],
    receivers: List[ReceiverHead],
    true_id: int,
    false_id: int,
    device: torch.device,
    meta: Dict,
) -> Optional[Dict]:
    """
    Compute path_forward / path_reverse / path_combined for EVERY
    (sender, receiver) pair for one (attack, example), reusing the
    per-sender hybrid encoder state and per-(sender, receiver-layer)
    Step-B cache across receivers as described in the module docstring.

    Returns None (and the caller should skip/record this as filtered)
    when |score_attack - score_control| < SKIP_EPSILON, matching the
    project-wide successful-instance filter (delta > 1e-4). Otherwise
    returns {"score_control", "score_attack", "delta", "rows": [...]}
    with one row dict per (sender, receiver).
    """
    n_heads_enc, d_kv_enc, _ = enc_hooks.head_geometry(model)
    n_heads_dec, d_kv_dec, _ = dec_hooks.head_geometry(model)
    enc_full_mask = enc_hooks.block_diag_head_mask(n_heads_enc, d_kv_enc, device)
    dec_full_mask = dec_hooks.block_diag_head_mask(n_heads_dec, d_kv_dec, device)

    sender_layers = sorted({s.layer for s in senders})
    receivers_by_layer = group_receivers_by_layer(receivers)
    receiver_layers = sorted(receivers_by_layer.keys())

    ctrl_o_cache, ctrl_hidden, ctrl_mask, score_control = cache_encoder_and_score(
        model, control_enc, device, sender_layers, true_id, false_id
    )
    atk_o_cache, atk_hidden, atk_mask, score_attack = cache_encoder_and_score(
        model, attack_enc, device, sender_layers, true_id, false_id
    )

    delta = score_attack - score_control
    if abs(delta) < SKIP_EPSILON:
        return None

    rows: List[Dict] = []
    for sender in senders:
        L_e, h_e = sender.layer, sender.head_idx
        single_mask = enc_full_mask[h_e : h_e + 1]

        hybrid_fwd_hidden, hybrid_fwd_mask = build_encoder_hybrid(
            model, control_enc, atk_o_cache[enc_hooks.slot_key(L_e)], L_e, single_mask, device
        )
        hybrid_rev_hidden, hybrid_rev_mask = build_encoder_hybrid(
            model, attack_enc, ctrl_o_cache[enc_hooks.slot_key(L_e)], L_e, single_mask, device
        )

        for L_r in receiver_layers:
            heads_here = receivers_by_layer[L_r]

            step_b_fwd = compute_receiver_layer_step_b_cache(
                model, ctrl_hidden, ctrl_mask, hybrid_fwd_hidden, L_r, true_id, false_id, device
            )
            fwd_scores = head_scores_for_slot(
                model, ctrl_hidden, ctrl_mask, "decoder_cross_attn", L_r, step_b_fwd, dec_full_mask,
                true_id, false_id,
            )

            step_b_rev = compute_receiver_layer_step_b_cache(
                model, atk_hidden, atk_mask, hybrid_rev_hidden, L_r, true_id, false_id, device
            )
            rev_scores = head_scores_for_slot(
                model, atk_hidden, atk_mask, "decoder_cross_attn", L_r, step_b_rev, dec_full_mask,
                true_id, false_id,
            )

            for recv in heads_here:
                h_r = recv.head_idx
                score_path_fwd = fwd_scores[h_r]
                score_path_rev = rev_scores[h_r]
                path_forward = (score_path_fwd - score_control) / delta
                path_reverse = (score_attack - score_path_rev) / delta
                path_combined = min(path_forward, path_reverse)
                rows.append({
                    "attack_name": meta["attack_name"],
                    "qid": meta["qid"],
                    "docid": meta["docid"],
                    "sample_index": meta.get("sample_index"),
                    "seed": meta.get("seed"),
                    "sender_layer": L_e,
                    "sender_head": h_e,
                    "sender_name": sender.label,
                    "receiver_layer": L_r,
                    "receiver_head": h_r,
                    "receiver_name": recv.label,
                    "score_control": score_control,
                    "score_attack": score_attack,
                    "delta": delta,
                    "score_path_forward": score_path_fwd,
                    "score_path_reverse": score_path_rev,
                    "path_forward": path_forward,
                    "path_reverse": path_reverse,
                    "path_combined": path_combined,
                })

    return {"score_control": score_control, "score_attack": score_attack, "delta": delta, "rows": rows}
