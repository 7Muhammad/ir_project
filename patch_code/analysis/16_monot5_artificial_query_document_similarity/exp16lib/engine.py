"""
exp16lib/engine.py
===================
Batched forward passes: ONE forward per batch yields all 25 checkpoint
similarities (hooks in exp16lib.checkpoints reduce each state immediately to
one cosine per sequence). Optionally also the monoT5 score of the same
forward (one decoder step from decoder_start_token_id; score =
logit("true") - logit("false"), identical to Exp 01's scoring).

Sequences are sorted by length inside `run_sequences` to minimise batch
padding; results are returned in input order.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch

from exp16lib.checkpoints import SimilarityCapture
from exp16lib.inputs import EncodedSeq, collate


@torch.inference_mode()
def forward_batch(model, batch, with_score: bool = False, true_id: int = None, false_id: int = None,
                  keep_states: bool = False):
    """Returns (sims [B, C] float64 tensor, scores [B] or None, capture)."""
    with SimilarityCapture(model.encoder, keep_states=keep_states) as cap:
        cap.set_masks(batch["query_mask"], batch["doc_mask"])
        if with_score:
            B = batch["input_ids"].shape[0]
            dec = torch.full((B, 1), model.config.decoder_start_token_id, dtype=torch.long,
                             device=batch["input_ids"].device)
            logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                           decoder_input_ids=dec).logits[:, 0, :]
            scores = (logits[:, true_id] - logits[:, false_id]).double()
        else:
            model.encoder(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            scores = None
        sims = cap.similarities()
    return sims, scores, cap


def run_sequences(model, seqs: List[EncodedSeq], pad_id: int, device: torch.device, batch_size: int,
                  with_score: bool = False, true_id: int = None, false_id: int = None
                  ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Similarities [N, C] (and scores [N]) for a list of sequences, in input order."""
    for s in seqs:
        if s.n_query_tokens < 1 or s.n_doc_tokens < 1:
            raise ValueError("every sequence needs >= 1 query and >= 1 document token")
    order = sorted(range(len(seqs)), key=lambda i: seqs[i].seq_len)
    sims_out: List[Optional[np.ndarray]] = [None] * len(seqs)
    score_out: List[Optional[float]] = [None] * len(seqs)
    for start in range(0, len(order), batch_size):
        idx = order[start:start + batch_size]
        batch = collate([seqs[i] for i in idx], pad_id, device)
        sims, scores, _ = forward_batch(model, batch, with_score, true_id, false_id)
        sims = sims.cpu().numpy()
        if not np.isfinite(sims).all():
            raise FloatingPointError("non-finite similarity")
        for j, i in enumerate(idx):
            sims_out[i] = sims[j]
            if scores is not None:
                score_out[i] = float(scores[j])
    S = np.stack(sims_out)
    return S, (np.array(score_out, dtype=np.float64) if with_score else None)
