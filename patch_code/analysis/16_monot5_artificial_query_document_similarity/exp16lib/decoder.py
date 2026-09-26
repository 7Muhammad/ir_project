"""
exp16lib/decoder.py
====================
Decoder-side analogue of the encoder query/document similarity.

monoT5's decoder runs ONE position (decoder_start_token_id), so it has no
query or document tokens of its own; it sees them only through cross-attention
to the (final-normed) encoder output. For every decoder layer L we therefore
split that layer's cross-attention output (after `.o`, before the residual
add) by SOURCE span:

    z_S[h]   = sum_{j in S} P[h, j] V[h, j]          (P: cross-attn weights, V = v(enc_out))
    m_S      = o( concat_h z_S[h] )                   (768-d message read from span S)
    msg_cos(L) = cos(m_query, m_doc)

Because `.o` is linear, m_query + m_doc + m_rest == the module's attention
output exactly (m_rest = template / EOS positions); this is checked on every
batch as a RELATIVE error, max|sum - output| / max|output| (`decomposition_max_rel`;
outputs reach ~2.5e3 in late layers, so an absolute tolerance would only
measure fp32 rounding). Masked positions (padded-control insertion
slots, batch padding) have P = 0 and contribute nothing. Span masks are the
encoder-analysis masks (query text only; document incl. injected attack tokens
for attacked inputs, active document positions only for the control).

Hook (transformers 5.9.0): forward hook with kwargs on
decoder.block[L].layer[1].EncDecAttention (T5Attention with
key_value_states); P = output[2] when the model is called with
output_attentions=True. Also stored per layer: cross-attention mass on the
query and document spans (mean over heads) and the two message norms.
"""

from __future__ import annotations

from typing import Dict, List

import torch

from exp16lib.pooling import cosine

N_DECODER_LAYERS = 12


def decoder_checkpoint_names(n_layers: int = N_DECODER_LAYERS) -> List[str]:
    return [f"D{L:02d}_cross_attn" for L in range(n_layers)]


DECODER_CHECKPOINTS = decoder_checkpoint_names()
FIELDS = ["msg_cos", "mass_query", "mass_doc", "norm_query_msg", "norm_doc_msg"]


def head_values(module, kv: torch.Tensor) -> torch.Tensor:
    """Per-head value vectors of a T5Attention for key/value source kv [B, S, d_model] -> [B, H, S, d_kv] (fp64)."""
    B, S, _ = kv.shape
    return module.v(kv).view(B, S, module.n_heads, module.key_value_proj_dim).transpose(1, 2).to(torch.float64)


def span_head_z(P: torch.Tensor, V: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """z_S[b, h] = sum_{j in S} P[b, h, j] V[b, h, j]  (P [B,H,S], V [B,H,S,dk], mask [B,S]) -> [B, H, dk]."""
    return torch.einsum("bhs,bs,bhsk->bhk", P, mask, V)


class CrossAttnMessageCapture:
    """Context manager; call set_masks([B,S] query mask, [B,S] doc mask) before each forward."""

    def __init__(self, decoder):
        self.decoder = decoder
        self.n_layers = len(decoder.block)
        self._handles = []
        self._q = self._d = None
        self.results: Dict[int, Dict[str, torch.Tensor]] = {}
        self.head_cos: Dict[int, torch.Tensor] = {}
        self.decomposition_max_rel = 0.0

    def set_masks(self, query_mask: torch.Tensor, doc_mask: torch.Tensor) -> None:
        self._q, self._d = query_mask, doc_mask
        self.results = {}
        self.head_cos = {}
        self.decomposition_max_rel = 0.0

    def _hook(self, L: int):
        def fn(module, args, kwargs, output):
            kv = kwargs.get("key_value_states")
            if kv is None:
                raise RuntimeError("EncDecAttention called without key_value_states")
            if len(output) < 3:
                raise RuntimeError("attention weights missing: call the model with output_attentions=True")
            P = output[2][:, :, 0, :].to(torch.float64)                      # [B, H, S]
            B = kv.shape[0]
            H, dk = module.n_heads, module.key_value_proj_dim
            V = head_values(module, kv)                                        # [B, H, S, dk]
            Wo = module.o.weight.to(torch.float64)                            # [d_model, H*dk]
            q, d = self._q.to(torch.float64), self._d.to(torch.float64)
            rest = 1.0 - q - d
            zq, zd, zr = (span_head_z(P, V, m) for m in (q, d, rest))         # [B, H, dk] each (pre-.o)

            def msg(z):
                return z.reshape(B, H * dk) @ Wo.T                            # [B, d_model]

            mq, md, mr = msg(zq), msg(zd), msg(zr)
            # per-head query- vs document-sourced 64-d contributions (Exp 16 head analysis)
            self.head_cos[L] = cosine(zq.reshape(B * H, dk), zd.reshape(B * H, dk)).reshape(B, H)
            full = output[0][:, 0, :].to(torch.float64)
            rel = float((mq + md + mr - full).abs().max() / full.abs().max().clamp_min(1e-12))
            self.decomposition_max_rel = max(self.decomposition_max_rel, rel)
            self.results[L] = {
                "msg_cos": cosine(mq, md),
                "mass_query": (P * q[:, None, :]).sum(-1).mean(1),
                "mass_doc": (P * d[:, None, :]).sum(-1).mean(1),
                "norm_query_msg": mq.norm(dim=-1),
                "norm_doc_msg": md.norm(dim=-1),
            }
        return fn

    def __enter__(self):
        for L, blk in enumerate(self.decoder.block):
            self._handles.append(blk.layer[1].EncDecAttention.register_forward_hook(self._hook(L), with_kwargs=True))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles = []
        return False

    def stacked(self) -> Dict[str, torch.Tensor]:
        missing = [L for L in range(self.n_layers) if L not in self.results]
        if missing:
            raise RuntimeError(f"decoder layers not captured: {missing}")
        return {f: torch.stack([self.results[L][f] for L in range(self.n_layers)], 1) for f in FIELDS}


@torch.inference_mode()
def decoder_forward_batch(model, batch, true_id: int, false_id: int):
    """Returns ({field: [B, n_layers] float64}, scores [B], decomposition_max_rel)."""
    B = batch["input_ids"].shape[0]
    dec = torch.full((B, 1), model.config.decoder_start_token_id, dtype=torch.long, device=batch["input_ids"].device)
    with CrossAttnMessageCapture(model.decoder) as cap:
        cap.set_masks(batch["query_mask"], batch["doc_mask"])
        out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], decoder_input_ids=dec,
                    output_attentions=True, use_cache=False)
        res = cap.stacked()
        err = cap.decomposition_max_rel
    logits = out.logits[:, 0, :]
    return res, (logits[:, true_id] - logits[:, false_id]).double(), err


def run_decoder_sequences(model, seqs, pad_id: int, device, batch_size: int, true_id: int, false_id: int, rtol: float):
    """Length-sorted batches; returns ({field: np [N, n_layers]}, scores np [N], worst relative decomposition error)."""
    import numpy as np
    from exp16lib.inputs import collate

    order = sorted(range(len(seqs)), key=lambda i: seqs[i].seq_len)
    outs = {f: [None] * len(seqs) for f in FIELDS}
    scores = [None] * len(seqs)
    worst = 0.0
    for start in range(0, len(order), batch_size):
        idx = order[start:start + batch_size]
        res, sc, err = decoder_forward_batch(model, collate([seqs[i] for i in idx], pad_id, device), true_id, false_id)
        worst = max(worst, err)
        if err > rtol:
            raise AssertionError(f"cross-attention message decomposition relative error {err} > {rtol}")
        for f in FIELDS:
            arr = res[f].cpu().numpy()
            if not np.isfinite(arr).all():
                raise FloatingPointError(f"non-finite {f}")
            for j, i in enumerate(idx):
                outs[f][i] = arr[j]
        for j, i in enumerate(idx):
            scores[i] = float(sc[j])
    return {f: np.stack(v) for f, v in outs.items()}, np.array(scores), worst
