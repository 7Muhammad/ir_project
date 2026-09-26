"""
exp16lib/decoder_probe.py
==========================
Decoder residual-stream analogue of the encoder query/document similarity.

monoT5's decoder runs one position (decoder_start_token_id); it sees the
query and document only through cross-attention. To get "the decoder's
representation of the query" and "... of the document" at every decoder
sublayer, the encoder is run ONCE normally and the first decoder step is then
run twice from that same encoder output:

  query-only run : cross-attention may attend ONLY to query-text positions
  doc-only run   : cross-attention may attend ONLY to document positions
                   (for attacked inputs this includes the injected tokens)

(implemented by passing `encoder_outputs` and the span mask as the
cross-attention `attention_mask`; the encoder itself is unchanged). At every
decoder checkpoint c:

    dec_sim(c) = cos( h_c[query-only run], h_c[doc-only run] )

Checkpoints (transformers 5.9.0 T5Block, decoder: layer[0] self-attn,
layer[1] cross-attn, layer[2] FF), 1 + 3*12 = 37:
  dec_embedding          pre-hook on decoder.block[0] (start-token embedding)
  D{L}_post_self_attn    decoder.block[L].layer[0] output[0]  (residual after self-attn)
  D{L}_post_cross_attn   decoder.block[L].layer[1] output[0]  (residual after cross-attn)
  D{L}_post_mlp          decoder.block[L].layer[2] output     (residual after FFN; L11 PRE final_layer_norm)

This is a PROBE: restricting cross-attention renormalises the attention over
the kept span, so each run is a hypothetical decoder pass, not the model's
normal run. By construction dec_embedding and D00_post_self_attn are
identical in both runs (cos = 1): nothing has read the encoder yet.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch
from transformers.modeling_outputs import BaseModelOutput

from exp16lib.pooling import cosine

N_DECODER_LAYERS = 12


def decoder_probe_checkpoints(n_layers: int = N_DECODER_LAYERS) -> List[str]:
    names = ["dec_embedding"]
    for L in range(n_layers):
        names += [f"D{L:02d}_post_self_attn", f"D{L:02d}_post_cross_attn", f"D{L:02d}_post_mlp"]
    return names


DECODER_PROBE_CHECKPOINTS = decoder_probe_checkpoints()
assert len(DECODER_PROBE_CHECKPOINTS) == 37


class DecoderStateCapture:
    """Records the single-position decoder residual state [B, d_model] at every checkpoint."""

    def __init__(self, decoder):
        self.decoder = decoder
        self.names = decoder_probe_checkpoints(len(decoder.block))
        self._handles = []
        self.states: Dict[int, torch.Tensor] = {}

    def _put(self, i, h):
        if h.dim() != 3 or h.shape[1] != 1:
            raise RuntimeError(f"{self.names[i]}: expected [B, 1, d_model], got {tuple(h.shape)}")
        self.states[i] = h[:, 0, :]

    def __enter__(self):
        self.states = {}

        def pre(module, args, kwargs):
            self._put(0, args[0] if args else kwargs["hidden_states"])

        self._handles.append(self.decoder.block[0].register_forward_pre_hook(pre, with_kwargs=True))
        for L, blk in enumerate(self.decoder.block):
            base = 1 + 3 * L
            self._handles.append(blk.layer[0].register_forward_hook(lambda m, a, o, _i=base: self._put(_i, o[0])))
            self._handles.append(blk.layer[1].register_forward_hook(lambda m, a, o, _i=base + 1: self._put(_i, o[0])))
            self._handles.append(blk.layer[2].register_forward_hook(lambda m, a, o, _i=base + 2: self._put(_i, o)))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles = []
        return False

    def stacked(self) -> torch.Tensor:
        """[B, n_checkpoints, d_model]"""
        missing = [self.names[i] for i in range(len(self.names)) if i not in self.states]
        if missing:
            raise RuntimeError(f"decoder checkpoints not captured: {missing}")
        return torch.stack([self.states[i] for i in range(len(self.names))], 1)


@torch.inference_mode()
def decoder_probe_batch(model, batch, keep_states: bool = False):
    """Returns (sims [B, 37] float64, extras). One encoder pass, one decoder pass over 2B probe rows."""
    ids, am = batch["input_ids"], batch["attention_mask"]
    qm, dm = batch["query_mask"], batch["doc_mask"]
    if bool((qm.sum(1) < 1).any()) or bool((dm.sum(1) < 1).any()):
        raise ValueError("empty query or document span")
    if bool(((qm > 0) & (am == 0)).any()) or bool(((dm > 0) & (am == 0)).any()):
        raise ValueError("span mask selects an attention-masked position")
    B = ids.shape[0]
    enc = model.encoder(input_ids=ids, attention_mask=am).last_hidden_state
    enc2 = torch.cat([enc, enc], 0)
    cross_mask = torch.cat([qm, dm], 0).to(am.dtype)                # query-only rows, then doc-only rows
    dec = torch.full((2 * B, 1), model.config.decoder_start_token_id, dtype=torch.long, device=ids.device)
    with DecoderStateCapture(model.decoder) as cap:
        out = model(encoder_outputs=BaseModelOutput(last_hidden_state=enc2), attention_mask=cross_mask,
                    decoder_input_ids=dec, use_cache=False)
        H = cap.stacked()                                            # [2B, C, d]
    hq, hd = H[:B], H[B:]
    C = H.shape[1]
    sims = cosine(hq.reshape(B * C, -1), hd.reshape(B * C, -1)).reshape(B, C)
    extras = {"logits": out.logits[:, 0, :]}
    if keep_states:
        extras["hq"], extras["hd"], extras["enc"] = hq, hd, enc
    return sims, extras


def run_decoder_probe(model, seqs, pad_id: int, device, batch_size: int):
    """Length-sorted batches -> np [N, 37]."""
    from exp16lib.inputs import collate

    order = sorted(range(len(seqs)), key=lambda i: seqs[i].seq_len)
    out = np.full((len(seqs), len(DECODER_PROBE_CHECKPOINTS)), np.nan)
    for start in range(0, len(order), batch_size):
        idx = order[start:start + batch_size]
        sims, _ = decoder_probe_batch(model, collate([seqs[i] for i in idx], pad_id, device))
        out[idx] = sims.cpu().numpy()
    if not np.isfinite(out).all():
        raise FloatingPointError("non-finite decoder probe similarity")
    return out
