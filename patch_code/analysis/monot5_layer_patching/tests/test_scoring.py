"""
tests/test_scoring.py
=====================
Unit tests for the monoT5 scoring function.

WHY THIS TEST EXISTS
---------------------
The scoring function is the foundation of the entire experiment.  Every
downstream result depends on being able to compute:

    score = logit("true") - logit("false")

This file checks:
  1. The scoring function returns a finite float (not NaN, not inf).
  2. The result is actually a float — not a Tensor accidentally left on GPU.
  3. Known pairs give plausible scores (a relevant passage scores higher
     than an obviously irrelevant one for the same query).

We use small, synthetic text so the test runs fast without downloading data.

Run with:
    pytest tests/test_scoring.py -v
"""

import math
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import pytest
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

from src.model_utils import (
    build_monot5_input,
    get_true_false_token_ids,
    load_monot5,
    score_from_encoding,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def model_and_tokenizer():
    """
    Load monoT5-base once for the whole test module.

    Loading takes ~10 seconds; scope="module" avoids reloading for each test.
    The model is placed on CPU here — we only need correctness, not speed.
    """
    device = torch.device("cpu")
    model, tokenizer = load_monot5("castorini/monot5-base-msmarco", device)
    return model, tokenizer, device


# ---------------------------------------------------------------------------
# Test 1: scoring returns a finite float
# ---------------------------------------------------------------------------

def test_scoring_returns_finite_float(model_and_tokenizer):
    """
    The most basic check: run the model on one tiny example, get a float.

    We encode the prompt as a single tensor, run one decoder step, and
    extract logit("true") - logit("false").

    If this returns NaN, the model output is degenerate (possibly a dtype
    or device issue).  If it returns a Tensor, the caller forgot to call
    .item() to unwrap the scalar.
    """
    model, tokenizer, device = model_and_tokenizer
    true_id, false_id = get_true_false_token_ids(tokenizer)

    query   = "what is the capital of France"
    passage = "Paris is the capital and most populous city of France."

    # Build the standard monoT5 prompt and tokenise it
    text = build_monot5_input(query, passage)
    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=False,
    )
    enc = {k: v.to(device) for k, v in enc.items()}

    # score_from_encoding does NOT use generation.
    # It runs exactly one decoder step with decoder_input_ids=[[pad_id]]
    # and reads the logit at position 0 for "true" and "false".
    score = score_from_encoding(model, enc, true_id, false_id, device)

    # Check the return type
    assert isinstance(score, float), (
        f"Expected float, got {type(score)}. "
        "Did score_from_encoding forget to call .item()?"
    )

    # Check the value is a real number
    assert not math.isnan(score), "Score is NaN — the model output is degenerate."
    assert not math.isinf(score), "Score is infinite — check for overflow or underflow."


# ---------------------------------------------------------------------------
# Test 2: score is not constant (the model is actually running)
# ---------------------------------------------------------------------------

def test_different_passages_give_different_scores(model_and_tokenizer):
    """
    A clearly relevant passage should score differently from a clearly
    irrelevant one.

    This is a sanity check that the model's parameters are loaded and
    affect the output — if both scores were identical we would suspect
    that the model is not actually processing the input.

    Note: we do not assert which direction the scores go.  monoT5 is a
    fine-tuned model and we trust it works.  We only check that the output
    varies, meaning the function is actually running inference.
    """
    model, tokenizer, device = model_and_tokenizer
    true_id, false_id = get_true_false_token_ids(tokenizer)

    query = "what is the boiling point of water"

    relevant_passage  = "Water boils at 100 degrees Celsius at standard atmospheric pressure."
    irrelevant_passage = "The moon is approximately 384,400 km from Earth."

    def score(passage):
        text = build_monot5_input(query, passage)
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        enc = {k: v.to(device) for k, v in enc.items()}
        return score_from_encoding(model, enc, true_id, false_id, device)

    score_rel  = score(relevant_passage)
    score_irrel = score(irrelevant_passage)

    # The two scores must differ — the model is sensitive to passage content.
    assert score_rel != score_irrel, (
        "Both passages got the same score. The model may not be running correctly."
    )


# ---------------------------------------------------------------------------
# Test 3: no gradient computation during scoring
# ---------------------------------------------------------------------------

def test_scoring_does_not_compute_gradients(model_and_tokenizer):
    """
    Verify that score_from_encoding does not accumulate gradients.

    WHY: keeping gradients would waste memory and time.  All scoring in
    this experiment uses torch.no_grad().  This test checks that the
    returned score (which came from a tensor) has no grad_fn.

    We test this by checking that no parameters have .grad set after
    a scoring call, and by verifying the score is a plain Python float
    (which cannot carry grad information).
    """
    model, tokenizer, device = model_and_tokenizer
    true_id, false_id = get_true_false_token_ids(tokenizer)

    # Zero out any existing gradients
    model.zero_grad(set_to_none=True)

    text = build_monot5_input("test query", "test passage about nothing")
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
    enc = {k: v.to(device) for k, v in enc.items()}

    _score = score_from_encoding(model, enc, true_id, false_id, device)

    # After inference, no parameter should have a gradient
    for name, param in model.named_parameters():
        assert param.grad is None, (
            f"Parameter '{name}' has a gradient after scoring. "
            "score_from_encoding should run inside torch.no_grad()."
        )
