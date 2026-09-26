"""
exp16lib/checkpoints.py
========================
The 25 primary encoder checkpoints and the hooks that read them.

Hook points (transformers 5.9.0, models/t5/modeling_t5.py, verified by
reading the installed source):

  T5Stack.forward:   hidden_states = self.dropout(inputs_embeds)
                     for layer_module in self.block: hidden_states = layer_module(hidden_states, ...)[0]
                     hidden_states = self.final_layer_norm(hidden_states)       <- NOT a primary checkpoint
  T5Block.forward:   hidden_states = self.layer[0](hidden_states, ...)[0]       # T5LayerSelfAttention
                     hidden_states = self.layer[-1](hidden_states)              # T5LayerFF
  T5LayerSelfAttention.forward:
                     hidden_states = hidden_states + self.dropout(SelfAttention(layer_norm(hidden_states))[0])
  T5LayerFF.forward: hidden_states = hidden_states + self.dropout(DenseReluDense(layer_norm(hidden_states)))

Therefore
  embedding       = forward-PRE-hook on encoder.block[0], positional arg 0
                    (= dropout(embed_tokens(input_ids)); dropout is identity
                    in eval mode, T5 does not scale embeddings) — exactly the
                    tensor entering encoder layer 0.
  L{L}_post_attn  = forward hook on encoder.block[L].layer[0], output[0]:
                    residual stream AFTER the self-attention residual update,
                    before the FFN sublayer.
  L{L}_post_mlp   = forward hook on encoder.block[L].layer[-1], output:
                    residual stream AFTER the FFN residual update = the
                    block output passed to layer L+1. For L=11 this is
                    taken BEFORE encoder.final_layer_norm.

All are [batch, seq_len, d_model]. For every checkpoint the hook immediately
reduces the state to the pooled query/document cosine (one scalar per
sequence), so no full hidden-state tensor is kept unless `keep_states=True`
(used only by sanity checks and tests).
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch

from exp16lib.pooling import pooled_cosine

N_ENCODER_LAYERS = 12


def checkpoint_names(n_layers: int = N_ENCODER_LAYERS) -> List[str]:
    names = ["embedding"]
    for L in range(n_layers):
        names += [f"L{L:02d}_post_attn", f"L{L:02d}_post_mlp"]
    return names


CHECKPOINTS: List[str] = checkpoint_names(N_ENCODER_LAYERS)
assert len(CHECKPOINTS) == 25


def checkpoint_table(n_layers: int = N_ENCODER_LAYERS) -> List[Dict]:
    """checkpoint_index, checkpoint_name, layer (-1 = embedding), sublayer."""
    rows = []
    for i, name in enumerate(checkpoint_names(n_layers)):
        if name == "embedding":
            rows.append({"checkpoint_index": i, "checkpoint_name": name, "layer": -1, "sublayer": "embedding"})
        else:
            rows.append({"checkpoint_index": i, "checkpoint_name": name, "layer": int(name[1:3]),
                         "sublayer": "attn" if name.endswith("post_attn") else "mlp"})
    return rows


class SimilarityCapture:
    """
    Context manager: registers the 1 + 2*n_layers hooks on `encoder` (a
    T5Stack). Before each forward call `set_masks(query_mask, doc_mask)`
    ([B, S] 0/1 tensors on the model's device); after it, `similarities()`
    returns a [B, n_checkpoints] float64 tensor in checkpoint order.
    """

    def __init__(self, encoder, keep_states: bool = False):
        self.encoder = encoder
        self.n_layers = len(encoder.block)
        self.names = checkpoint_names(self.n_layers)
        self.keep_states = keep_states
        self._handles = []
        self._q: Optional[torch.Tensor] = None
        self._d: Optional[torch.Tensor] = None
        self._sims: Dict[int, torch.Tensor] = {}
        self.states: Dict[str, torch.Tensor] = {}

    def set_masks(self, query_mask: torch.Tensor, doc_mask: torch.Tensor) -> None:
        self._q, self._d = query_mask, doc_mask
        self._sims = {}
        self.states = {}

    def _record(self, idx: int, h: torch.Tensor) -> None:
        if self._q is None:
            raise RuntimeError("SimilarityCapture.set_masks was not called before forward")
        if h.dim() != 3 or h.shape[:2] != self._q.shape:
            raise RuntimeError(f"checkpoint {self.names[idx]}: state shape {tuple(h.shape)} "
                               f"incompatible with masks {tuple(self._q.shape)}")
        self._sims[idx] = pooled_cosine(h, self._q, self._d)
        if self.keep_states:
            self.states[self.names[idx]] = h.detach().clone()

    def __enter__(self):
        enc = self.encoder

        def pre_embed(module, args, kwargs):
            h = args[0] if args else kwargs["hidden_states"]
            self._record(0, h)

        self._handles.append(enc.block[0].register_forward_pre_hook(pre_embed, with_kwargs=True))
        for L, block in enumerate(enc.block):
            def post_attn(module, args, output, _i=1 + 2 * L):
                self._record(_i, output[0])

            def post_mlp(module, args, output, _i=2 + 2 * L):
                self._record(_i, output)

            self._handles.append(block.layer[0].register_forward_hook(post_attn))
            self._handles.append(block.layer[-1].register_forward_hook(post_mlp))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles = []
        return False

    def similarities(self) -> torch.Tensor:
        n = len(self.names)
        missing = [self.names[i] for i in range(n) if i not in self._sims]
        if missing:
            raise RuntimeError(f"checkpoints not captured: {missing}")
        return torch.stack([self._sims[i] for i in range(n)], dim=1)
