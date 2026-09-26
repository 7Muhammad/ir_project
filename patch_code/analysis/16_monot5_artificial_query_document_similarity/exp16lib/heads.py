"""
exp16lib/heads.py
==================
Head-level query/document similarity at the canonical important heads.

Head sets (imported, never redefined): Exp 13's
  exp13lib.head_lists.load_senders()   -> 18 encoder self-attention heads
  exp13lib.head_lists.load_receivers() -> 31 decoder CROSS-attention heads
                                          (the loader asserts component == "decoder_cross_attn")
Heads are ordered by (layer, head).

Encoder head h in layer L (Exp 11 head geometry): the input of
encoder.block[L].layer[0].SelfAttention.o is concat_h(head_h) with head h in
the slice [h*d_kv, (h+1)*d_kv). A forward-PRE-hook on `.o` reads that 64-d
per-position head output (pre-o_proj, i.e. before heads are mixed) and
    enc_head_cos(h) = cos( mean_{query-text positions} head_h,
                           mean_{document positions}   head_h )
with the Exp 16 masks (exp16lib.pooling.pooled_cosine).

Decoder head h in layer L (first decoder step only): with the cross-attention
weights P[h, j] of the single decoder position and V[h, j] = v(enc_out)_h,
    z_query(h) = sum_{j in query positions}    P[h, j] V[h, j]   (64-d)
    z_doc(h)   = sum_{j in document positions} P[h, j] V[h, j]   (64-d)
    dec_head_cos(h) = cos(z_query(h), z_doc(h))
computed by exp16lib.decoder.CrossAttnMessageCapture (same split whose
layer-level `.o` projection is exactness-checked on every batch).
"""

from __future__ import annotations

from typing import Dict, List

import torch

import exp16lib  # noqa: F401
from exp13lib.head_lists import load_receivers, load_senders
from exp16lib.decoder import CrossAttnMessageCapture
from exp16lib.pooling import pooled_cosine


def encoder_heads():
    return sorted(load_senders(), key=lambda h: (h.layer, h.head_idx))


def decoder_heads():
    return sorted(load_receivers(), key=lambda h: (h.layer, h.head_idx))


class EncoderHeadCapture:
    """Pre-hooks on SelfAttention.o of the layers hosting the given heads; call set_masks before forward."""

    def __init__(self, encoder, heads, keep_inputs: bool = False):
        self.encoder, self.heads = encoder, heads
        self.layers = sorted({h.layer for h in heads})
        self.keep_inputs = keep_inputs
        self._handles = []
        self._q = self._d = None
        self.cos: Dict[str, torch.Tensor] = {}
        self.inputs: Dict[int, torch.Tensor] = {}

    def set_masks(self, query_mask, doc_mask):
        self._q, self._d = query_mask, doc_mask
        self.cos, self.inputs = {}, {}

    def _hook(self, L):
        def fn(module, args):
            inner = args[0]                                          # [B, S, H*d_kv], pre-o_proj
            dk = self.encoder.config.d_kv
            if self.keep_inputs:
                self.inputs[L] = inner.detach().clone()
            for h in self.heads:
                if h.layer == L:
                    self.cos[h.label] = pooled_cosine(inner[..., h.head_idx * dk:(h.head_idx + 1) * dk],
                                                      self._q, self._d)
        return fn

    def __enter__(self):
        for L in self.layers:
            self._handles.append(self.encoder.block[L].layer[0].SelfAttention.o.register_forward_pre_hook(self._hook(L)))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles = []
        return False


@torch.inference_mode()
def head_forward_batch(model, batch, enc_heads, dec_heads):
    """One full forward (first decoder step) -> ({enc label: [B]}, {dec label: [B]}, decoder decomposition rel err)."""
    B = batch["input_ids"].shape[0]
    dec = torch.full((B, 1), model.config.decoder_start_token_id, dtype=torch.long, device=batch["input_ids"].device)
    with EncoderHeadCapture(model.encoder, enc_heads) as ec, CrossAttnMessageCapture(model.decoder) as dc:
        ec.set_masks(batch["query_mask"], batch["doc_mask"])
        dc.set_masks(batch["query_mask"], batch["doc_mask"])
        model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], decoder_input_ids=dec,
              output_attentions=True, use_cache=False)
        enc = {h.label: ec.cos[h.label] for h in enc_heads}
        decd = {h.label: dc.head_cos[h.layer][:, h.head_idx] for h in dec_heads}
        err = dc.decomposition_max_rel
    return enc, decd, err


def run_head_sequences(model, seqs, pad_id, device, batch_size, enc_heads, dec_heads, rtol: float):
    """Length-sorted batches -> (np [N, n_enc], np [N, n_dec], worst decoder decomposition rel err)."""
    import numpy as np
    from exp16lib.inputs import collate

    order = sorted(range(len(seqs)), key=lambda i: seqs[i].seq_len)
    E = np.full((len(seqs), len(enc_heads)), np.nan)
    D = np.full((len(seqs), len(dec_heads)), np.nan)
    worst = 0.0
    for start in range(0, len(order), batch_size):
        idx = order[start:start + batch_size]
        enc, decd, err = head_forward_batch(model, collate([seqs[i] for i in idx], pad_id, device), enc_heads, dec_heads)
        if err > rtol:
            raise AssertionError(f"decoder decomposition relative error {err} > {rtol}")
        worst = max(worst, err)
        e = torch.stack([enc[h.label] for h in enc_heads], 1).cpu().numpy()
        d = torch.stack([decd[h.label] for h in dec_heads], 1).cpu().numpy()
        E[idx], D[idx] = e, d
    if not (np.isfinite(E).all() and np.isfinite(D).all()):
        raise FloatingPointError("non-finite head similarity")
    return E, D, worst


def head_labels(heads) -> List[str]:
    return [h.label for h in heads]
