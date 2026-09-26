"""
src/decoderlens.py
==================
The DecoderLens core: read the attack signal off each encoder layer.

Normal monoT5 scoring uses ONLY the final encoder output:

    final encoder hidden states -> decoder -> logit(true), logit(false)

DecoderLens instead forces the decoder to cross-attend to EACH encoder layer's
hidden states in turn:

    encoder layer 0 hidden states  -> decoder -> score_0
    encoder layer 1 hidden states  -> decoder -> score_1
    ...
    encoder layer 12 hidden states -> decoder -> score_12

with  score_l = logit("true") - logit("false").

Why apply the encoder final layer norm to each layer?
-----------------------------------------------------
T5's encoder returns ``hidden_states`` as a tuple
(embedding_output, block_1_out, ..., block_12_out).  These are the RAW block
outputs BEFORE the encoder's final layer norm.  The model's actual
``last_hidden_state`` (what the decoder normally sees) is
``final_layer_norm(block_12_out)``.  So to put every intermediate layer on the
same footing the decoder expects — and to make layer 12 match the normal
monoT5 score — we apply ``model.encoder.final_layer_norm`` to each hidden state
before handing it to the decoder.

This is NOT activation patching: nothing is swapped between two forward passes.
We only re-read a single forward pass's intermediate encoder representations.

Everything runs under model.eval() + torch.no_grad() with dropout disabled.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers.modeling_outputs import BaseModelOutput


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def resolve_device(device_cfg: str) -> torch.device:
    """Resolve 'auto' | 'cuda' | 'cpu' into a torch.device."""
    if device_cfg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_cfg)


def load_monot5(
    checkpoint: str, device: torch.device
) -> Tuple[T5ForConditionalGeneration, T5Tokenizer]:
    """Load monoT5 + tokenizer, place in eval mode (dropout disabled)."""
    print(f"[decoderlens] Loading tokenizer: {checkpoint}")
    # use_fast=False forces the SentencePiece slow tokenizer (no protobuf needed).
    tokenizer = T5Tokenizer.from_pretrained(checkpoint, use_fast=False)
    print(f"[decoderlens] Loading model: {checkpoint}")
    model = T5ForConditionalGeneration.from_pretrained(checkpoint).to(device)
    model.eval()
    print(f"[decoderlens] Loaded on device: {device}")
    return model, tokenizer


def get_true_false_token_ids(tokenizer: T5Tokenizer) -> Tuple[int, int]:
    """Return (true_id, false_id); raise if either is not a single token."""
    true_tokens = tokenizer.encode("true", add_special_tokens=False)
    false_tokens = tokenizer.encode("false", add_special_tokens=False)
    if len(true_tokens) != 1:
        raise ValueError(
            f"'true' tokenises to {len(true_tokens)} tokens ({true_tokens}); "
            "monoT5 scoring requires exactly 1."
        )
    if len(false_tokens) != 1:
        raise ValueError(
            f"'false' tokenises to {len(false_tokens)} tokens ({false_tokens}); "
            "monoT5 scoring requires exactly 1."
        )
    return true_tokens[0], false_tokens[0]


def num_encoder_layers(model: T5ForConditionalGeneration) -> int:
    """
    Number of DecoderLens layers = encoder blocks + 1 (the embedding layer).

    For monoT5-base this is 12 + 1 = 13 (indices 0..12).
    """
    return model.config.num_layers + 1


# ---------------------------------------------------------------------------
# DecoderLens core
# ---------------------------------------------------------------------------

def get_encoder_layer_outputs(
    model: T5ForConditionalGeneration,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> List[torch.Tensor]:
    """
    Run the encoder once and return every layer's hidden state.

    HuggingFace T5 ``encoder.hidden_states`` layout (length = num_blocks + 1)
    -----------------------------------------------------------------------
        index 0          -> embedding / pre-block-1 representation
        index l (1..N-1) -> output of encoder block l (RAW, pre final layer norm)
        index N          -> output of encoder block N, ALREADY passed through the
                            encoder final layer norm (this IS last_hidden_state)

    So the TOP entry is already normalised, while every lower entry is a raw
    block output.  Callers must therefore apply ``final_layer_norm`` to the
    lower entries but NOT to the top entry (doing so would double-normalise it
    and break the final-layer sanity check against the normal monoT5 score).

    Each tensor has shape (batch, seq_len, d_model).
    """
    with torch.no_grad():
        enc = model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
    return list(enc.hidden_states)


def score_with_encoder_layer(
    model: T5ForConditionalGeneration,
    encoder_hidden_state: torch.Tensor,
    attention_mask: torch.Tensor,
    decoder_input_ids: torch.Tensor,
    true_id: int,
    false_id: int,
    apply_final_layer_norm: bool = True,
) -> Tuple[float, float, float]:
    """
    Score one encoder hidden state by cross-attending the decoder to it.

    Parameters
    ----------
    encoder_hidden_state : Tensor (1, L, d_model) — a single encoder layer's
        hidden states (RAW block output).
    attention_mask : Tensor (1, L) — the SAME mask used for the encoder pass so
        padded / inserted positions (mask=0) are ignored by cross-attention.
    decoder_input_ids : Tensor (1, 1) — the T5 decoder start token.
    apply_final_layer_norm : bool — apply model.encoder.final_layer_norm so that
        the top layer matches the normal monoT5 score.

    Returns
    -------
    (score, true_logit, false_logit) where score = true_logit - false_logit.
    """
    h = encoder_hidden_state
    if apply_final_layer_norm:
        h = model.encoder.final_layer_norm(h)
        # T5 applies dropout after the final layer norm during training; in
        # eval mode dropout is a no-op, but we call it for exact parity.
        h = model.encoder.dropout(h)

    encoder_outputs = BaseModelOutput(last_hidden_state=h)
    with torch.no_grad():
        outputs = model(
            encoder_outputs=encoder_outputs,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
        )
    logits = outputs.logits[0, 0]  # (vocab,)
    true_logit = logits[true_id].item()
    false_logit = logits[false_id].item()
    return (true_logit - false_logit), true_logit, false_logit


def layerwise_scores_for_input(
    model: T5ForConditionalGeneration,
    enc: Dict[str, torch.Tensor],
    true_id: int,
    false_id: int,
    apply_final_layer_norm: bool = True,
) -> List[Tuple[float, float, float]]:
    """
    Compute DecoderLens scores for one encoder input across ALL encoder layers.

    One encoder forward pass + one (cheap, single-token) decoder pass per layer.

    Returns a list of (score, true_logit, false_logit), one per encoder layer,
    indexed 0..num_encoder_layers-1.
    """
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    decoder_input_ids = torch.tensor(
        [[model.config.decoder_start_token_id]], device=input_ids.device
    )

    hidden_states = get_encoder_layer_outputs(model, input_ids, attention_mask)
    n = len(hidden_states)
    results: List[Tuple[float, float, float]] = []
    for i, h_l in enumerate(hidden_states):
        # The top entry is already final-layer-normalised by the encoder; every
        # lower entry is a raw block output that must be normalised here so all
        # layers are on the footing the decoder expects.
        is_last = i == n - 1
        apply_ln = apply_final_layer_norm and not is_last
        results.append(
            score_with_encoder_layer(
                model=model,
                encoder_hidden_state=h_l,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                true_id=true_id,
                false_id=false_id,
                apply_final_layer_norm=apply_ln,
            )
        )
    return results


def normal_monot5_score(
    model: T5ForConditionalGeneration,
    enc: Dict[str, torch.Tensor],
    true_id: int,
    false_id: int,
) -> float:
    """
    The standard monoT5 score using the full encoder->decoder forward pass.

    Used only for the final-layer sanity check (it should match the
    DecoderLens score at the top encoder layer).
    """
    decoder_input_ids = torch.tensor(
        [[model.config.decoder_start_token_id]], device=enc["input_ids"].device
    )
    with torch.no_grad():
        outputs = model(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            decoder_input_ids=decoder_input_ids,
        )
    logits = outputs.logits[0, 0]
    return (logits[true_id] - logits[false_id]).item()
