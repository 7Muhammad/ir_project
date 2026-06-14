"""
src/scoring.py
==============
monoT5 relevance scoring — no generation, one decoder step only.

Three input variants per (query, passage) pair
-----------------------------------------------
A. original_score  : score(original clean input — no prefix)
   Used for reference only.  NOT used in patching.

B. control_score   : score(padded control input)
   The attack prefix positions are filled with pad tokens (mask=0).
   This is the baseline for all patching comparisons.

C. attack_score    : score(attacked input — "relevant:×5" prefix)
   The adversarial keyword-stuffing input from the ECIR-24 paper.

Key metrics saved per pair
--------------------------
  control_delta_vs_original = control_score - original_score
      How much does the padded control (empty slots) shift the score relative
      to the original?  Should be small if pad slots are truly neutral.

  attack_delta_vs_original  = attack_score - original_score
      Overall attack strength relative to the unmodified document.

  attack_delta_vs_control   = attack_score - control_score
      The effect we isolate and explain via activation patching.
      Because control and attack have identical sequence length and
      passage alignment, this difference is purely due to what is in
      the attack-prefix token positions: real "relevant:" tokens (attack)
      vs empty pad slots (control).

Scoring procedure (one decoder step)
--------------------------------------
1. Tokenise the full prompt string (or use pre-built input_ids + mask).
2. Create decoder_input_ids = [[pad_token_id]] (T5 decoder start).
3. Run model forward once with no_grad().
4. Get logits at decoder position 0: shape (1, vocab_size).
5. score = logit("true") - logit("false").
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

from src.model_utils import (
    build_monot5_input,
    build_padded_control_and_attack_encodings,
    score_from_encoding,
)


# ---------------------------------------------------------------------------
# Text-based scoring (used only for original clean input — type A)
# ---------------------------------------------------------------------------

def score_single(
    model: T5ForConditionalGeneration,
    tokenizer: T5Tokenizer,
    input_text: str,
    true_id: int,
    false_id: int,
    max_length: int,
    device: torch.device,
) -> float:
    """
    Score a single text prompt (type A — original clean input).

    Parameters
    ----------
    model : T5ForConditionalGeneration (eval mode)
    tokenizer : T5Tokenizer
    input_text : full monoT5 prompt string
    true_id, false_id : int
    max_length : int
    device : torch.device

    Returns
    -------
    float : logit("true") - logit("false")
    """
    enc = tokenizer(
        input_text,
        return_tensors="pt",
        max_length=max_length,
        truncation=True,
        padding=False,
    ).to(device)
    decoder_input_ids = torch.tensor(
        [[model.config.decoder_start_token_id]], device=device
    )
    with torch.no_grad():
        outputs = model(**enc, decoder_input_ids=decoder_input_ids)
    logits = outputs.logits[0, 0]
    return (logits[true_id] - logits[false_id]).item()


def score_batch(
    model: T5ForConditionalGeneration,
    tokenizer: T5Tokenizer,
    input_texts: List[str],
    true_id: int,
    false_id: int,
    max_length: int,
    device: torch.device,
    batch_size: int = 8,
) -> List[float]:
    """
    Score a list of text prompts in batches (used for original clean inputs).

    Parameters
    ----------
    model, tokenizer, true_id, false_id, max_length, device : see score_single
    input_texts : list of prompt strings
    batch_size : int

    Returns
    -------
    List of floats, same length as input_texts.
    """
    scores = []
    for start in range(0, len(input_texts), batch_size):
        batch_texts = input_texts[start: start + batch_size]
        enc = tokenizer(
            batch_texts,
            return_tensors="pt",
            max_length=max_length,
            truncation=True,
            padding=True,  # pad to longest in batch
        ).to(device)
        batch_sz = enc["input_ids"].shape[0]
        decoder_input_ids = torch.full(
            (batch_sz, 1),
            model.config.decoder_start_token_id,
            dtype=torch.long,
            device=device,
        )
        with torch.no_grad():
            outputs = model(**enc, decoder_input_ids=decoder_input_ids)
        logits = outputs.logits[:, 0, :]  # (batch, vocab_size)
        batch_scores = (logits[:, true_id] - logits[:, false_id]).tolist()
        scores.extend(batch_scores)
    return scores


# ---------------------------------------------------------------------------
# Score all three variants for a list of pairs
# ---------------------------------------------------------------------------

def score_all_variants(
    pairs: List[Dict],
    model: T5ForConditionalGeneration,
    tokenizer: T5Tokenizer,
    true_id: int,
    false_id: int,
    max_length: int,
    device: torch.device,
    batch_size: int = 8,
) -> List[Dict]:
    """
    For each (query, passage) pair compute all three monoT5 scores.

    The three input types are:
      A. original clean   — text-based, batched (no attack)
      B. padded control   — token-level, per-example (pad slots in attack positions)
      C. attacked input   — token-level, per-example (real attack tokens from repo)

    Each pair dict must contain:
        query            — query text
        passage          — original clean passage (text_0 from attacked TSV)
        attacked_passage — pre-built attacked passage (text from attacked TSV)

    WHY ORIGINAL IS BATCHED BUT CONTROL/ATTACK ARE PER-EXAMPLE
    ------------------------------------------------------------
    Type A (original) is a plain text input — all examples can be batched
    with standard tokenizer padding.

    Types B and C are built at the token level with a custom attention_mask
    (zeros for pad slots in B).  Each example produces a sequence of a
    different total length (different queries and passages), so within-batch
    padding would add another layer of padding on top of the structural pad
    slots in B.  To avoid that complexity, we score B and C one example at
    a time (batch_size=1).

    Metrics saved per pair:
      original_score           — reference, not used in patching
      control_score            — padded control baseline
      attack_score             — attacked input (from repo)
      control_delta_vs_original = control_score - original_score
      attack_delta_vs_original  = attack_score  - original_score
      attack_delta_vs_control   = attack_score  - control_score
          ↑ This is the quantity that activation patching tries to explain.

    Parameters
    ----------
    pairs : list of dicts with "query", "passage", "attacked_passage"
    model, tokenizer, true_id, false_id, max_length, device
    batch_size : int
        Used only for original-clean batched scoring.

    Returns
    -------
    List of dicts with all score fields added.
    """
    # --- A. Score original clean inputs (batched, text-based) ---
    original_texts = [
        build_monot5_input(p["query"], p["passage"])
        for p in pairs
    ]
    print(f"[scoring] Scoring {len(pairs)} original (clean) inputs ...")
    original_scores = score_batch(
        model, tokenizer, original_texts,
        true_id, false_id, max_length, device, batch_size
    )

    # --- B & C. Score padded control and attacked inputs (per-example) ---
    # Each example gets its own padded control built at the token level.
    # The attacked passage comes directly from the pair dict (loaded from repo).
    print(f"[scoring] Scoring {len(pairs)} padded-control + attacked inputs ...")
    control_scores: List[float] = []
    attack_scores:  List[float] = []

    for i, pair in enumerate(pairs):
        if (i + 1) % 50 == 0:
            print(f"[scoring]   {i+1}/{len(pairs)}")

        # Both padded control (B) and attacked (C) are built together.
        # This validates passage alignment and raises a clear error if it fails.
        control_enc, attack_enc, n_prefix, n_attack_tokens = (
            build_padded_control_and_attack_encodings(
                tokenizer=tokenizer,
                query=pair["query"],
                passage=pair["passage"],
                attacked_passage=pair["attacked_passage"],
                max_length=max_length,
                device=device,
            )
        )
        ctrl_score = score_from_encoding(model, control_enc, true_id, false_id, device)
        atk_score  = score_from_encoding(model, attack_enc,  true_id, false_id, device)

        control_scores.append(ctrl_score)
        attack_scores.append(atk_score)

    # --- Assemble results ---
    results: List[Dict] = []
    for pair, orig, ctrl, atk in zip(pairs, original_scores, control_scores, attack_scores):
        r = dict(pair)
        r["original_score"]            = orig
        r["control_score"]             = ctrl   # padded control score
        r["attack_score"]              = atk
        # Deltas
        r["control_delta_vs_original"] = ctrl - orig  # pad-slot baseline shift
        r["attack_delta_vs_original"]  = atk  - orig  # total attack strength
        r["attack_delta_vs_control"]   = atk  - ctrl  # what patching explains
        results.append(r)

    return results


# ---------------------------------------------------------------------------
# Example selection
# ---------------------------------------------------------------------------

def select_attacked_examples(
    scored_pairs: List[Dict],
    min_attack_delta: float,
    max_selected: int,
    seed: int = 42,
) -> List[Dict]:
    """
    Select examples where the attack raised the score above the padded control.

    Selection criterion:
        attack_score - control_score > min_attack_delta

    Using control (padded) as the baseline rather than original ensures we
    select examples where the attack tokens specifically increase the score,
    not just examples where the attacked input is generically longer.

    If more examples pass than max_selected, keep the top-N by delta.

    Parameters
    ----------
    scored_pairs : output of score_all_variants
    min_attack_delta : float  (0.0 keeps all examples where attack > control)
    max_selected : int
    seed : int  (kept for reproducibility signature; not used in top-N sort)

    Returns
    -------
    List of selected scored pair dicts.
    """
    selected = [
        p for p in scored_pairs
        if p["attack_delta_vs_control"] > min_attack_delta
    ]
    # Sort by delta descending — most clearly attacked examples first
    selected.sort(key=lambda x: x["attack_delta_vs_control"], reverse=True)

    if len(selected) > max_selected:
        selected = selected[:max_selected]

    print(
        f"[scoring] Selected {len(selected)} examples "
        f"(attack_delta_vs_control > {min_attack_delta}) "
        f"from {len(scored_pairs)} total."
    )
    return selected
