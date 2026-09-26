"""
exp16lib/sanity.py
===================
Model-level checks that the 25 checkpoints mean what DECISIONS 23-28 say.
Run on the tiny random T5 in the unit tests and on real monoT5 at the start
of stage 01 (fail loudly). All checks use ONE encoder pass with extra
independent hooks + output_hidden_states=True:

  embedding    == embed_tokens(input_ids) == hidden_states[0]
  post_attn[L] == block_input[L] + SelfAttention(...)[0]    (attention residual update)
  post_mlp[L]  == post_attn[L] + DenseReluDense(layer_norm(post_attn[L]))  (FFN residual update)
  post_mlp[L]  == block_input[L+1] == hidden_states[L+1]     (state carried forward), L < last
  final_layer_norm(post_mlp[last]) == last_hidden_state      (post_mlp[last] is PRE final norm)
  every state is [B, S, d_model]; hook similarities == pooled_cosine(recorded state)
"""

from __future__ import annotations

from typing import Dict, List

import torch

from exp16lib.checkpoints import SimilarityCapture
from exp16lib.inputs import EncodedSeq, collate
from exp16lib.pooling import pooled_cosine


def _maxdiff(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.double() - b.double()).abs().max())


@torch.inference_mode()
def hook_semantics_report(model, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
    enc = model.encoder
    n_layers = len(enc.block)
    d_model = model.config.d_model
    block_in: Dict[int, torch.Tensor] = {}
    attn_out: Dict[int, torch.Tensor] = {}
    handles = []
    for L, blk in enumerate(enc.block):
        handles.append(blk.register_forward_pre_hook(
            lambda m, a, k, _L=L: block_in.__setitem__(_L, (a[0] if a else k["hidden_states"]).clone()),
            with_kwargs=True))
        handles.append(blk.layer[0].SelfAttention.register_forward_hook(
            lambda m, a, o, _L=L: attn_out.__setitem__(_L, o[0].clone())))
    try:
        with SimilarityCapture(enc, keep_states=True) as cap:
            cap.set_masks(batch["query_mask"], batch["doc_mask"])
            out = enc(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                      output_hidden_states=True)
            sims = cap.similarities()
            st = cap.states
    finally:
        for h in handles:
            h.remove()

    B, S = batch["input_ids"].shape
    rep: Dict[str, float] = {"n_checkpoints": float(len(st))}
    rep["shape_ok"] = float(all(tuple(v.shape) == (B, S, d_model) for v in st.values()))
    hs = out.hidden_states
    rep["embedding_vs_embed_tokens"] = _maxdiff(st["embedding"], enc.embed_tokens(batch["input_ids"]))
    rep["embedding_vs_hidden_states0"] = _maxdiff(st["embedding"], hs[0])
    e_attn, e_mlp, e_carry = 0.0, 0.0, 0.0
    for L in range(n_layers):
        pa, pm = st[f"L{L:02d}_post_attn"], st[f"L{L:02d}_post_mlp"]
        e_attn = max(e_attn, _maxdiff(pa, block_in[L] + attn_out[L]))
        ff = enc.block[L].layer[-1]
        e_mlp = max(e_mlp, _maxdiff(pm, pa + ff.DenseReluDense(ff.layer_norm(pa))))
        if L + 1 < n_layers:
            e_carry = max(e_carry, _maxdiff(pm, block_in[L + 1]), _maxdiff(pm, hs[L + 1]))
    last = st[f"L{n_layers - 1:02d}_post_mlp"]
    rep["post_attn_residual"] = e_attn
    rep["post_mlp_residual"] = e_mlp
    rep["post_mlp_carried_forward"] = e_carry
    rep["final_norm_of_last_post_mlp_vs_encoder_output"] = _maxdiff(enc.final_layer_norm(last), out.last_hidden_state)
    # How different the (excluded) final-normalised state is: must be clearly non-zero.
    rep["last_post_mlp_minus_encoder_output"] = _maxdiff(last, out.last_hidden_state)
    recomputed = torch.stack([pooled_cosine(st[n], batch["query_mask"], batch["doc_mask"]) for n in cap.names], 1)
    rep["hook_sims_vs_recomputed"] = _maxdiff(sims, recomputed)
    rep["post_attn_differs_from_raw_attention_output"] = max(
        _maxdiff(st[f"L{L:02d}_post_attn"], attn_out[L]) for L in range(n_layers))
    return rep


def assert_hook_semantics(model, batch, atol: float, expected_checkpoints: int) -> Dict[str, float]:
    rep = hook_semantics_report(model, batch)
    errs = []
    if int(rep["n_checkpoints"]) != expected_checkpoints:
        errs.append(f"n_checkpoints={rep['n_checkpoints']} != {expected_checkpoints}")
    if rep["shape_ok"] != 1.0:
        errs.append("state shape is not [B, S, d_model]")
    for k in ("embedding_vs_embed_tokens", "embedding_vs_hidden_states0", "post_attn_residual", "post_mlp_residual",
              "post_mlp_carried_forward", "final_norm_of_last_post_mlp_vs_encoder_output", "hook_sims_vs_recomputed"):
        if rep[k] > atol:
            errs.append(f"{k}={rep[k]:.3g} > {atol}")
    for k in ("last_post_mlp_minus_encoder_output", "post_attn_differs_from_raw_attention_output"):
        if rep[k] <= atol:
            errs.append(f"{k}={rep[k]:.3g} should be clearly non-zero")
    if errs:
        raise AssertionError("hook semantics check failed: " + "; ".join(errs))
    return rep


@torch.inference_mode()
def batch_invariance_report(model, seqs: List[EncodedSeq], pad_id: int, device: torch.device) -> float:
    """max |sim(batched with padding) - sim(alone)| over the given sequences and all checkpoints."""
    from exp16lib.engine import forward_batch
    joint, _, _ = forward_batch(model, collate(seqs, pad_id, device))
    worst = 0.0
    for i, s in enumerate(seqs):
        alone, _, _ = forward_batch(model, collate([s], pad_id, device))
        worst = max(worst, _maxdiff(joint[i], alone[0]))
    return worst
