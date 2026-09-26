"""
src/model_utils.py
==================
Loads monoT5 (T5ForConditionalGeneration) and its tokenizer from HuggingFace.
Also provides the padded-control tokenisation logic for activation patching.

How monoT5 scoring works
-------------------------
monoT5 is a T5 model fine-tuned as a pointwise relevance classifier.
It takes a natural-language prompt like:

    "Query: {query} Document: {passage} Relevant:"

and is trained to predict whether the continuation is "true" (relevant) or
"false" (not relevant).

At inference time we do NOT run generation.  Instead:
  1. Encode the prompt with the T5 encoder.
  2. Create a single-token decoder input containing just the T5 decoder-start
     token (<pad>, id=0).
  3. Run one decoder step to get the logit distribution over the vocabulary.
  4. Extract logit("true") and logit("false").
  5. Compute score = logit("true") - logit("false").

Why logit("true") - logit("false")?
    The model was trained to answer "true" or "false".  The raw logit for
    "true" measures how strongly the model leans toward relevance.  However,
    the overall calibration can vary.  Taking the difference cancels that
    shared additive offset and gives a calibrated score: positive means
    "relevant", negative means "not relevant".  This follows the original
    monoT5 paper and the Pygaggle library implementation.

Three input types used in this experiment
-----------------------------------------
A. Original clean reference:
       "Query: {query} Document: {passage} Relevant:"
   Used only for reference scoring — NOT used in activation patching.
   The original clean and attacked inputs have different sequence lengths,
   so their hidden-state tensors have different shapes and cannot be
   directly swapped.

B. Padded control (input type B):
   Constructed at the token level — not as a text string.
   The attack-prefix positions are filled with tokenizer.pad_token_id
   and masked with attention_mask=0 so the encoder ignores them.
   Passage tokens appear at the SAME absolute position as in the attacked input.

       input_ids      = [prefix_tokens] + [pad_id × N] + [passage + EOS]
       attention_mask = [1  × prefix  ] + [0  × N    ] + [1 × suffix  ]

C. Attacked input (input type C):
       "Query: {query} Document: {attacked_passage} Relevant:"
   where attacked_passage is the pre-built attacked text from the repo
   (e.g. "relevant: relevant: relevant: relevant: relevant: {original_passage}").
   All tokens are real; attention_mask=1 everywhere.

       input_ids      = [prefix_tokens] + [attack_tokens × N] + [passage + EOS]
       attention_mask = [1  × prefix  ] + [1 × N            ] + [1 × suffix  ]

Why padded control instead of a neutral-word control?
------------------------------------------------------
A neutral-word prefix (e.g. "information:" × 5) also solves the
sequence-length problem but introduces another meaningful word that could
independently shift the monoT5 score.  The padded control uses
zero-attention pad slots — the encoder sees those positions as empty —
giving a cleaner baseline that isolates the attack tokens' effect.

Why must padded control and attacked input have the same length?
---------------------------------------------------------------
Full-layer activation patching substitutes an entire hidden-state tensor of
shape (batch, seq_len, d_model) from one forward pass into another.  If
seq_len differs, the shapes do not match and the substitution fails.  By
inserting exactly N pad tokens (N = number of tokens the attack contributes),
both sequences have the same length:
    n_prefix + N + len(passage_tokens + EOS)
and passage token j is at the same absolute index in both sequences.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer


def resolve_device(device_cfg: str) -> torch.device:
    """
    Resolve the device string from the config into a torch.device.

    Parameters
    ----------
    device_cfg : str
        "auto" → GPU if available, else CPU.
        "cuda"  → GPU (raises if unavailable).
        "cpu"   → CPU.
    """
    if device_cfg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_cfg)


def load_monot5(
    checkpoint: str,
    device: torch.device,
) -> Tuple[T5ForConditionalGeneration, T5Tokenizer]:
    """
    Load monoT5 model and tokenizer from HuggingFace Hub.

    The model is placed in evaluation mode (model.eval()) immediately.
    Always wrap inference calls with torch.no_grad().

    Parameters
    ----------
    checkpoint : str
        HuggingFace model ID, e.g. "castorini/monot5-base-msmarco".
    device : torch.device

    Returns
    -------
    (model, tokenizer)
    """
    print(f"[model_utils] Loading tokenizer from: {checkpoint}")
    # use_fast=False forces the slow (SentencePiece-based) tokenizer.
    # The "fast" tokenizer requires protobuf to convert the .spiece.model file,
    # which may not be installed.  The slow tokenizer uses sentencepiece directly
    # and is functionally identical for T5 inference.
    tokenizer: T5Tokenizer = T5Tokenizer.from_pretrained(checkpoint, use_fast=False)

    print(f"[model_utils] Loading model from: {checkpoint}")
    model: T5ForConditionalGeneration = T5ForConditionalGeneration.from_pretrained(
        checkpoint
    )
    model = model.to(device)
    model.eval()  # disable dropout — we do inference only
    print(f"[model_utils] Model loaded on device: {device}")
    return model, tokenizer


def get_true_false_token_ids(tokenizer: T5Tokenizer) -> Tuple[int, int]:
    """
    Return the token IDs for the single tokens "true" and "false" in T5.

    monoT5 scoring requires that "true" and "false" are each a SINGLE token.
    If the tokenizer splits either word into sub-tokens, the score computation
    would be ambiguous.

    Returns
    -------
    (true_id, false_id)

    Raises
    ------
    ValueError if either word is not a single token.
    """
    true_tokens  = tokenizer.encode("true",  add_special_tokens=False)
    false_tokens = tokenizer.encode("false", add_special_tokens=False)

    if len(true_tokens) != 1:
        raise ValueError(
            f"'true' tokenises to {len(true_tokens)} tokens ({true_tokens}), "
            "but monoT5 scoring requires exactly 1 token for 'true'."
        )
    if len(false_tokens) != 1:
        raise ValueError(
            f"'false' tokenises to {len(false_tokens)} tokens ({false_tokens}), "
            "but monoT5 scoring requires exactly 1 token for 'false'."
        )

    return true_tokens[0], false_tokens[0]


def build_monot5_input(query: str, passage: str) -> str:
    """
    Construct the ORIGINAL CLEAN monoT5 prompt (input type A — no prefix).

    This is the unmodified document format used for reference scoring only.
    It has a different sequence length than the attacked input and is NOT
    used in activation patching (shapes would not match).

    Format: "Query: {query} Document: {passage} Relevant:"
    """
    return f"Query: {query} Document: {passage} Relevant:"


def build_padded_control_and_attack_encodings(
    tokenizer: T5Tokenizer,
    query: str,
    passage: str,
    attacked_passage: str,
    max_length: int,
    device: torch.device,
) -> Tuple[Dict, Dict, int, int]:
    """
    Build tokenised padded-control (B) and attacked (C) encoder inputs.

    WHY WE USE PRE-BUILT ATTACKED PASSAGE
    ----------------------------------------
    The ECIR-24 repo stores the exact attacked passages used in the original
    paper (runs/injected/dl19/relevant_start_5_bm25_19.gz.tsv).  We load
    those directly so our experiment operates on the same attacked text, not
    a re-constructed version that might differ due to SentencePiece boundaries.

    WHY WE WORK AT THE TOKEN LEVEL
    --------------------------------
    We cannot build the padded control as a text string because inserting a
    placeholder string would be tokenised by the model's SentencePiece
    tokenizer and might produce unintended tokens.  Instead:

      1. Tokenise the full attacked prompt   → attacked_ids
      2. Tokenise the full original prompt   → original_ids
      3. Walk both sequences from the start to find the shared prefix length
         n_prefix (tokens before the attack tokens diverge from the passage).
      4. Compute N = len(attacked_ids) - len(original_ids) = number of tokens
         the attack contributes.
      5. Validate that original_ids[n_prefix:] == attacked_ids[n_prefix + N:]
         — i.e., the passage tokens are at the same relative position.
      6. Build control_ids by replacing attacked_ids[n_prefix:n_prefix+N]
         with [pad_token_id] × N.
      7. Set attention_mask=0 for those N pad slots.

    WHY attention_mask=0 FOR PAD SLOTS
    ------------------------------------
    T5's encoder self-attention adds a large negative value to the logits for
    positions where attention_mask=0, effectively preventing any token from
    attending to those positions.  The pad-slot positions are thus invisible
    to the rest of the sequence — a clean empty baseline.

    WHY PASSAGE ALIGNMENT MATTERS
    -------------------------------
    Activation patching swaps the ENTIRE (batch, seq_len, d_model) tensor
    between the control run and the attack run.  For this swap to be
    semantically valid, passage token j must be at the same absolute index
    in both sequences.  With N pad/attack tokens after the shared prefix,
    passage token j lives at index  n_prefix + N + j  in both sequences. ✓

    Parameters
    ----------
    tokenizer : T5Tokenizer
    query : str
    passage : str
        The original clean passage text (text_0 from the attacked TSV).
    attacked_passage : str
        The pre-built attacked passage text (text from the attacked TSV),
        e.g. "relevant: relevant: relevant: relevant: relevant: {passage}".
    max_length : int
        Maximum encoder sequence length.  Both sequences are truncated to this.
    device : torch.device

    Returns
    -------
    control_enc : dict {"input_ids": Tensor(1, L), "attention_mask": Tensor(1, L)}
    attack_enc  : dict {"input_ids": Tensor(1, L), "attention_mask": Tensor(1, L)}
    n_prefix : int   — shared prefix token count
    n_attack_tokens : int   — N, number of pad slots / attack token positions

    Raises
    ------
    ValueError
        If passage tokens are not aligned at the same positions in both
        sequences (would make patching semantically invalid).
    """
    pad_id = tokenizer.pad_token_id

    # Build the full monoT5 prompt strings.
    # original_text  — clean prompt, no attack tokens
    # attacked_text  — prompt using the pre-built attacked passage from the repo
    original_text = build_monot5_input(query, passage)
    attacked_text = build_monot5_input(query, attacked_passage)

    # Tokenise both with EOS added (add_special_tokens=True is the default).
    # We do NOT truncate here so we can find the exact split point first.
    original_ids: List[int] = tokenizer.encode(original_text, add_special_tokens=True)
    attacked_ids: List[int] = tokenizer.encode(attacked_text, add_special_tokens=True)

    # --- Step 1: find the shared prefix length ---
    # Walk both token sequences from the left until they differ.
    # The point of divergence is where "Document: " ends and either the
    # passage (original) or the attack tokens (attacked) begins.
    n_prefix = 0
    for o_tok, a_tok in zip(original_ids, attacked_ids):
        if o_tok == a_tok:
            n_prefix += 1
        else:
            break  # first divergence = start of attack tokens / passage

    # --- Step 2: number of attack tokens ---
    # The attacked sequence is longer by exactly the number of tokens that
    # the attack prepends (assuming passage tokens are identical at the same
    # relative offset, which we validate below).
    n_attack_tokens = len(attacked_ids) - len(original_ids)
    if n_attack_tokens <= 0:
        raise ValueError(
            f"The attacked passage contributed {n_attack_tokens} additional tokens "
            "compared to the original. The attacked_passage must be strictly longer "
            "than the original passage. Check that the correct attacked TSV is loaded."
        )

    # --- Step 3: validate passage alignment ---
    # After the shared prefix in original  → passage tokens + EOS
    # After the shared prefix + N in attacked → same passage tokens + EOS
    # If these differ, passage positions do not align and patching is invalid.
    original_suffix: List[int] = original_ids[n_prefix:]
    attacked_suffix: List[int] = attacked_ids[n_prefix + n_attack_tokens:]

    if original_suffix != attacked_suffix:
        raise ValueError(
            "Passage alignment check failed: the passage tokens do not appear "
            "at the same relative positions in the original vs attacked prompts.\n"
            f"  original_suffix[:8]  = {original_suffix[:8]}\n"
            f"  attacked_suffix[:8]  = {attacked_suffix[:8]}\n"
            "This means SentencePiece re-tokenised the passage differently when "
            "the attack tokens appear before it.  The attacked passage loaded from "
            "the repo should end with a space before the passage text to preserve "
            "the SentencePiece ▁ prefix on the first passage word."
        )

    # --- Step 4: build padded control input_ids ---
    # Replace the N attack-token positions with pad tokens.
    # Layout: [shared_prefix | pad×N | passage+EOS]
    control_ids: List[int] = (
        original_ids[:n_prefix]       # shared prefix (same as attacked)
        + [pad_id] * n_attack_tokens  # pad slots replacing attack tokens
        + original_ids[n_prefix:]     # passage + EOS (= original_suffix)
    )

    # Explicit length assertion — should always pass given the construction above
    assert len(control_ids) == len(attacked_ids), (
        f"BUG: control length {len(control_ids)} != attack length {len(attacked_ids)}"
    )

    # --- Step 5: build attention masks ---
    # Control: real tokens → 1,  pad slots → 0 (encoder ignores those positions).
    # Attack:  all real tokens → 1 everywhere.
    n_suffix = len(original_suffix)  # passage tokens + EOS
    control_mask: List[int] = [1] * n_prefix + [0] * n_attack_tokens + [1] * n_suffix
    attack_mask:  List[int] = [1] * len(attacked_ids)

    # --- Step 6: truncate to max_length ---
    # Both sequences have the same length, so truncating at the same index
    # preserves alignment.
    control_ids  = control_ids[:max_length]
    control_mask = control_mask[:max_length]
    attacked_ids = attacked_ids[:max_length]
    attack_mask  = attack_mask[:max_length]

    # --- Step 7: convert to PyTorch tensors ---
    control_enc: Dict = {
        "input_ids":      torch.tensor([control_ids],  dtype=torch.long, device=device),
        "attention_mask": torch.tensor([control_mask], dtype=torch.long, device=device),
    }
    attack_enc: Dict = {
        "input_ids":      torch.tensor([attacked_ids], dtype=torch.long, device=device),
        "attention_mask": torch.tensor([attack_mask],  dtype=torch.long, device=device),
    }

    return control_enc, attack_enc, n_prefix, n_attack_tokens


def score_from_encoding(
    model: T5ForConditionalGeneration,
    enc: Dict,
    true_id: int,
    false_id: int,
    device: torch.device,
) -> float:
    """
    Score a pre-tokenised encoder input (with an explicit attention_mask).

    Used for the padded control (B) and attacked (C) inputs, which are built
    at the token level and carry a custom attention_mask.  The original clean
    input (A) can still use the text-based score_single function.

    Parameters
    ----------
    model : T5ForConditionalGeneration (must be in eval mode)
    enc : dict with "input_ids" (1, L) and "attention_mask" (1, L)
    true_id, false_id : int
    device : torch.device

    Returns
    -------
    float : logit("true") - logit("false")
    """
    enc_on_device = {k: v.to(device) for k, v in enc.items()}
    # Single decoder start token — same as in score_single
    decoder_input_ids = torch.tensor(
        [[model.config.decoder_start_token_id]], device=device
    )
    with torch.no_grad():
        outputs = model(**enc_on_device, decoder_input_ids=decoder_input_ids)
    logits = outputs.logits[0, 0]  # shape: (vocab_size,)
    return (logits[true_id] - logits[false_id]).item()


def build_padded_control_and_attack_encodings_general(
    tokenizer: T5Tokenizer,
    query: str,
    passage: str,
    attacked_passage: str,
    max_length: int,
    device: torch.device,
) -> Tuple[Optional[Dict], Optional[Dict], "AlignmentResult"]:
    """
    Position-agnostic padded-control construction using token-level alignment.

    Works for start, end, and random injection attacks by diffing the full
    prompt token sequences with difflib.SequenceMatcher (see src/alignment.py).

    This is the general replacement for build_padded_control_and_attack_encodings,
    which assumes all attack tokens form a single contiguous block immediately
    after the shared prefix.  That assumption holds for start attacks but
    fails for end and random attacks.

    For ``relevant_start_5`` this function produces token-for-token identical
    encodings to the legacy function (verified by test_alignment_general.py).

    On alignment failure, returns ``(None, None, AlignmentResult(status="failed", ...))``.
    Callers should log the failure and skip the example rather than raising.

    Parameters
    ----------
    tokenizer      : T5Tokenizer
    query          : str   — query text
    passage        : str   — original clean passage (text_0 from attacked TSV)
    attacked_passage : str — pre-built attacked passage (text from attacked TSV)
    max_length     : int   — maximum encoder sequence length; both sequences
                             are truncated to this after alignment.
    device         : torch.device

    Returns
    -------
    control_enc : dict {"input_ids": Tensor(1,L), "attention_mask": Tensor(1,L)}
                  or None on failure.
    attack_enc  : dict {"input_ids": Tensor(1,L), "attention_mask": Tensor(1,L)}
                  or None on failure.
    result      : AlignmentResult  — always returned; check result.status == "ok".
    """
    from src.alignment import align_attack, AlignmentResult  # avoid circular import

    pad_id = tokenizer.pad_token_id

    original_text = build_monot5_input(query, passage)
    attacked_text = build_monot5_input(query, attacked_passage)

    original_ids: List[int] = tokenizer.encode(original_text, add_special_tokens=True)
    attacked_ids: List[int] = tokenizer.encode(attacked_text, add_special_tokens=True)

    result: AlignmentResult = align_attack(original_ids, attacked_ids)
    if result.status != "ok":
        return None, None, result

    # Build padded-control: copy attacked_ids, replacing inserted positions
    # with pad_token_id and masking them with attention_mask=0.
    inserted_set = set(result.inserted_positions)
    control_ids: List[int] = [
        pad_id if i in inserted_set else tok
        for i, tok in enumerate(attacked_ids)
    ]
    control_mask: List[int] = [
        0 if i in inserted_set else 1 for i in range(len(attacked_ids))
    ]
    attack_mask: List[int] = [1] * len(attacked_ids)

    # Truncate — both sequences have the same length so the same slice applies.
    control_ids  = control_ids[:max_length]
    control_mask = control_mask[:max_length]
    attacked_ids = attacked_ids[:max_length]
    attack_mask  = attack_mask[:max_length]

    control_enc: Dict = {
        "input_ids":      torch.tensor([control_ids],  dtype=torch.long, device=device),
        "attention_mask": torch.tensor([control_mask], dtype=torch.long, device=device),
    }
    attack_enc: Dict = {
        "input_ids":      torch.tensor([attacked_ids], dtype=torch.long, device=device),
        "attention_mask": torch.tensor([attack_mask],  dtype=torch.long, device=device),
    }
    return control_enc, attack_enc, result
