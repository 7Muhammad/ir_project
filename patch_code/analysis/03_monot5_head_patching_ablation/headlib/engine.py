"""
headlib/engine.py
=================
Per-head patching/ablation engine for Experiment 3.

Reuses from Experiment 1 (imported, not duplicated):
  - model loading / scoring formula      src/model_utils.py
  - padded-control (Type B) construction src/model_utils.py + src/alignment.py
  - fwd / rev / combined effect formulas src/patching.py (Eq. 4-6 in paper.tex),
    applied here per head instead of per layer-component
  - SKIP_EPSILON degenerate-example guard src/patching.py

Two performance tricks (both exact, verified by tests/):

1. Encoder-output reuse.  All interventions are on DECODER attention heads,
   so the encoder forward is identical across every patched run of the same
   input.  We run the encoder once per input and reuse its hidden states for
   every decoder pass (~50-100x fewer FLOPs than full forwards).

2. Batched-over-heads decoder passes.  To score all n_heads heads of one
   (layer, component) we run ONE decoder pass with a batch of n_heads
   identical rows, where row h has only head h's slice patched (see
   headlib/head_hooks.py).  Rows are independent, so this equals n_heads
   separate single-head passes.

Score bookkeeping per (head, example)
--------------------------------------
  fwd patch : base = padded control, replacement = attack head activation
              fwd_effect = (score_patched_fwd - score_control) / delta
  rev patch : base = attack, replacement = control head activation
              rev_effect = (score_attack - score_patched_rev) / delta
  combined  = min(fwd_effect, rev_effect)
  delta     = score_attack - score_control   (as in Experiment 1)

  zero ablation : head output set to the zero vector (W_o has no bias, so
                  the head's residual contribution is exactly zero).
  mean ablation : head output replaced by the padded-control (Type B)
                  activation for that head.

  NOTE: on the ATTACK base input, "mean ablation" and "reverse patching"
  are the same computation (attack run, control activation at that head),
  so score_ablated_mean == score_patched_rev in Grid A/B by construction.
  We compute it once and write it to both columns.  On the CLEAN base input
  (side-effect runs) mean ablation is a distinct computation.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers.modeling_outputs import BaseModelOutput

from src.patching import SKIP_EPSILON  # same degenerate-example guard as Exp 1

from headlib.head_hooks import (
    block_diag_head_mask,
    get_o_proj,
    head_geometry,
    make_cache_pre_hook,
    make_per_row_head_pre_hook,
    slot_key,
)


# ---------------------------------------------------------------------------
# Encoder-output reuse + decoder-only scoring
# ---------------------------------------------------------------------------

def compute_encoder_states(
    model: nn.Module,
    enc: Dict[str, torch.Tensor],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Run the encoder once for a pre-built encoding and return
    (last_hidden_state, attention_mask), both on `device`.

    The attention_mask must accompany the hidden states into every decoder
    pass: cross-attention uses it to mask the padded-control's empty slots.
    """
    enc_dev = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        out = model.encoder(
            input_ids=enc_dev["input_ids"],
            attention_mask=enc_dev["attention_mask"],
        )
    return out.last_hidden_state, enc_dev["attention_mask"]


def decoder_pass_scores(
    model: nn.Module,
    enc_hidden: torch.Tensor,
    enc_mask: torch.Tensor,
    true_id: int,
    false_id: int,
    batch_rows: int = 1,
) -> torch.Tensor:
    """
    Run one single-step decoder pass over precomputed encoder states and
    return score = logit("true") - logit("false") per batch row.

    `enc_hidden` / `enc_mask` are (1, L, d) / (1, L) and are expanded (view,
    no copy) to `batch_rows` identical rows.  Without hooks all rows give the
    same score; with a per-row head hook, row h scores head h's intervention.
    """
    dec = torch.full(
        (batch_rows, 1),
        model.config.decoder_start_token_id,
        dtype=torch.long,
        device=enc_hidden.device,
    )
    with torch.no_grad():
        out = model(
            encoder_outputs=BaseModelOutput(
                last_hidden_state=enc_hidden.expand(batch_rows, -1, -1)
            ),
            attention_mask=enc_mask.expand(batch_rows, -1),
            decoder_input_ids=dec,
            use_cache=False,
        )
    logits = out.logits[:, 0, :]
    return logits[:, true_id] - logits[:, false_id]


# ---------------------------------------------------------------------------
# Head-activation caching (one pass caches all slots AND yields the base score)
# ---------------------------------------------------------------------------

def cache_head_inputs(
    model: nn.Module,
    enc_hidden: torch.Tensor,
    enc_mask: torch.Tensor,
    slots: List[Tuple[str, int]],
    true_id: int,
    false_id: int,
) -> Tuple[Dict[str, torch.Tensor], float]:
    """
    Run one unpatched decoder pass, caching the ``.o`` input (concatenated
    head outputs, shape (1, 1, inner_dim)) at every slot.

    Returns (cache, base_score) — the base score comes for free from the
    same pass.
    """
    cache: Dict[str, torch.Tensor] = {}
    handles = []
    try:
        for comp, layer in slots:
            handles.append(
                get_o_proj(model, comp, layer).register_forward_pre_hook(
                    make_cache_pre_hook(cache, slot_key(comp, layer))
                )
            )
        score = decoder_pass_scores(model, enc_hidden, enc_mask, true_id, false_id, 1)[0].item()
    finally:
        for h in handles:
            h.remove()
    return cache, score


def head_scores_for_slot(
    model: nn.Module,
    enc_hidden: torch.Tensor,
    enc_mask: torch.Tensor,
    component: str,
    layer_idx: int,
    replacement: Optional[torch.Tensor],
    head_mask: torch.Tensor,
    true_id: int,
    false_id: int,
) -> List[float]:
    """
    Score every head of one (component, layer) slot in a single batched
    decoder pass.  Row h has only head h's slice replaced (or zeroed when
    `replacement` is None).

    Returns a list of n_heads floats (index = head_idx).
    """
    n_heads = head_mask.shape[0]
    handle = get_o_proj(model, component, layer_idx).register_forward_pre_hook(
        make_per_row_head_pre_hook(replacement, head_mask)
    )
    try:
        scores = decoder_pass_scores(
            model, enc_hidden, enc_mask, true_id, false_id, n_heads
        )
    finally:
        handle.remove()
    return scores.tolist()


# ---------------------------------------------------------------------------
# Per-example drivers
# ---------------------------------------------------------------------------

def run_grid_example(
    model: nn.Module,
    control_enc: Dict[str, torch.Tensor],
    attack_enc: Dict[str, torch.Tensor],
    slots: List[Tuple[str, int]],
    true_id: int,
    false_id: int,
    device: torch.device,
    meta: Dict,
    methods: List[str],
) -> Optional[List[Dict]]:
    """
    Grid A/B row computation for one example: fwd/rev patching + zero/mean
    ablation (attack base) for every head in `slots`.

    `meta` must carry qid, docid, attack_name, score_clean (the original
    clean score from the selection record — the clean input plays no role
    in grid patching, exactly as in Experiment 1).

    Returns one row dict per (slot, head), or None when
    |score_attack - score_control| < SKIP_EPSILON (same skip rule as Exp 1).
    """
    n_heads, d_kv, _ = head_geometry(model)
    head_mask = block_diag_head_mask(n_heads, d_kv, device)

    ctrl_hidden, ctrl_mask = compute_encoder_states(model, control_enc, device)
    atk_hidden, atk_mask = compute_encoder_states(model, attack_enc, device)

    ctrl_cache, control_score = cache_head_inputs(
        model, ctrl_hidden, ctrl_mask, slots, true_id, false_id
    )
    atk_cache, attack_score = cache_head_inputs(
        model, atk_hidden, atk_mask, slots, true_id, false_id
    )

    delta = attack_score - control_score
    if abs(delta) < SKIP_EPSILON:
        print(
            f"[engine] Skipping qid={meta['qid']} docid={meta['docid']}: "
            f"attack_delta_vs_control={delta:.6f} < epsilon={SKIP_EPSILON}"
        )
        return None

    rows: List[Dict] = []
    for comp, layer in slots:
        key = slot_key(comp, layer)

        fwd_scores = head_scores_for_slot(
            model, ctrl_hidden, ctrl_mask, comp, layer,
            atk_cache[key], head_mask, true_id, false_id,
        )
        # Reverse patch == mean ablation on the attack base (see module docstring).
        rev_scores = head_scores_for_slot(
            model, atk_hidden, atk_mask, comp, layer,
            ctrl_cache[key], head_mask, true_id, false_id,
        )
        if "zero" in methods:
            zero_scores = head_scores_for_slot(
                model, atk_hidden, atk_mask, comp, layer,
                None, head_mask, true_id, false_id,
            )
        else:
            zero_scores = [None] * n_heads

        for h in range(n_heads):
            fwd_effect = (fwd_scores[h] - control_score) / delta
            rev_effect = (attack_score - rev_scores[h]) / delta
            mean_score = rev_scores[h] if "mean" in methods else None
            rows.append({
                "qid":               meta["qid"],
                "docid":             meta["docid"],
                "attack_name":       meta["attack_name"],
                "layer":             layer,
                "component":         comp,
                "head_idx":          h,
                "score_clean":       meta.get("score_clean"),
                "score_control":     control_score,
                "score_attack":      attack_score,
                "score_patched_fwd": fwd_scores[h],
                "score_patched_rev": rev_scores[h],
                "score_ablated_zero": zero_scores[h],
                "score_ablated_mean": mean_score,
                "fwd_effect":        fwd_effect,
                "rev_effect":        rev_effect,
                "combined_effect":   min(fwd_effect, rev_effect),
                # score_before = score_attack for the grid runs
                "score_drop_zero":   (attack_score - zero_scores[h]) if zero_scores[h] is not None else None,
                "score_drop_mean":   (attack_score - mean_score) if mean_score is not None else None,
            })
    return rows


def run_clean_example(
    model: nn.Module,
    clean_enc: Dict[str, torch.Tensor],
    control_enc: Dict[str, torch.Tensor],
    slots: List[Tuple[str, int]],
    true_id: int,
    false_id: int,
    device: torch.device,
    meta: Dict,
    methods: List[str],
) -> List[Dict]:
    """
    Clean side-effect row computation for one example: zero/mean ablation of
    every head while scoring the CLEAN (Type A, unattacked) input.

    Mean ablation still injects the padded-control (Type B) head activation;
    decoder head activations are always (1, 1, inner_dim), so the control
    activation drops into the clean run with no length alignment needed.

    `meta` must carry qid, docid, attack_name, and (for reference columns)
    score_attack from the selection record.
    """
    n_heads, d_kv, _ = head_geometry(model)
    head_mask = block_diag_head_mask(n_heads, d_kv, device)

    clean_hidden, clean_mask = compute_encoder_states(model, clean_enc, device)
    clean_score = decoder_pass_scores(
        model, clean_hidden, clean_mask, true_id, false_id, 1
    )[0].item()

    if "mean" in methods:
        ctrl_hidden, ctrl_mask = compute_encoder_states(model, control_enc, device)
        ctrl_cache, control_score = cache_head_inputs(
            model, ctrl_hidden, ctrl_mask, slots, true_id, false_id
        )
    else:
        ctrl_cache, control_score = {}, meta.get("score_control")

    rows: List[Dict] = []
    for comp, layer in slots:
        key = slot_key(comp, layer)
        if "zero" in methods:
            zero_scores = head_scores_for_slot(
                model, clean_hidden, clean_mask, comp, layer,
                None, head_mask, true_id, false_id,
            )
        else:
            zero_scores = [None] * n_heads
        if "mean" in methods:
            mean_scores = head_scores_for_slot(
                model, clean_hidden, clean_mask, comp, layer,
                ctrl_cache[key], head_mask, true_id, false_id,
            )
        else:
            mean_scores = [None] * n_heads

        for h in range(n_heads):
            rows.append({
                "qid":               meta["qid"],
                "docid":             meta["docid"],
                "attack_name":       meta["attack_name"],
                "layer":             layer,
                "component":         comp,
                "head_idx":          h,
                "score_clean":       clean_score,
                "score_control":     control_score,
                "score_attack":      meta.get("score_attack"),
                "score_ablated_zero": zero_scores[h],
                "score_ablated_mean": mean_scores[h],
                # score_before = score_clean for the side-effect runs
                "score_drop_zero":   (clean_score - zero_scores[h]) if zero_scores[h] is not None else None,
                "score_drop_mean":   (clean_score - mean_scores[h]) if mean_scores[h] is not None else None,
            })
    return rows
