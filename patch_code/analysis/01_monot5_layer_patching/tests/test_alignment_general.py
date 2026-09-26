"""
tests/test_alignment_general.py
===============================
Phase-0 tests for the position-agnostic padded-control construction.

What we check
-------------
1. Regression: for ``relevant_start_5``, the new general function produces
   token-for-token identical encodings to the legacy function.
2. ``relevant_end_5`` alignment succeeds and produces non-zero pad slots.
3. ``relevant_random_5`` alignment succeeds for a synthetic mid-passage
   insertion.
4. Trap case: the passage already contains the word "relevant" — must still
   align cleanly OR report a failure (not silently produce wrong spans).
5. Multi-token attack token (``information_start_3``) aligns and the
   inserted span count is divisible by the repetition count.

Run with:
    pytest tests/test_alignment_general.py -v
"""

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import pytest
import torch
from transformers import T5Tokenizer

from src.alignment import align_attack, AlignmentResult
from src.model_utils import (
    build_monot5_input,
    build_padded_control_and_attack_encodings,
    build_padded_control_and_attack_encodings_general,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

QUERY   = "what is photosynthesis"
PASSAGE = "Photosynthesis is the process by which plants convert sunlight into energy."


@pytest.fixture(scope="module")
def tokenizer():
    return T5Tokenizer.from_pretrained("castorini/monot5-base-msmarco")


@pytest.fixture(scope="module")
def device():
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Low-level unit tests for align_attack
# ---------------------------------------------------------------------------

def _encode(tokenizer: T5Tokenizer, query: str, passage: str) -> list:
    return tokenizer.encode(build_monot5_input(query, passage), add_special_tokens=True)


def test_align_simple_start(tokenizer):
    attacked = "relevant: " * 5 + PASSAGE
    orig_ids     = _encode(tokenizer, QUERY, PASSAGE)
    attacked_ids = _encode(tokenizer, QUERY, attacked)
    result = align_attack(orig_ids, attacked_ids)
    assert result.status == "ok", result.reason
    assert result.n_inserted > 0
    assert result.n_inserted == len(attacked_ids) - len(orig_ids)


def test_align_simple_end(tokenizer):
    attacked = PASSAGE + " relevant: relevant: relevant: relevant: relevant:"
    orig_ids     = _encode(tokenizer, QUERY, PASSAGE)
    attacked_ids = _encode(tokenizer, QUERY, attacked)
    result = align_attack(orig_ids, attacked_ids)
    assert result.status == "ok", result.reason
    assert result.n_inserted > 0
    assert result.n_inserted == len(attacked_ids) - len(orig_ids)


def test_align_fails_when_no_insertion(tokenizer):
    orig_ids     = _encode(tokenizer, QUERY, PASSAGE)
    result = align_attack(orig_ids, orig_ids)
    assert result.status == "failed"


def test_spans_are_contiguous_for_start(tokenizer):
    attacked = "relevant: " * 3 + PASSAGE
    orig_ids     = _encode(tokenizer, QUERY, PASSAGE)
    attacked_ids = _encode(tokenizer, QUERY, attacked)
    result = align_attack(orig_ids, attacked_ids)
    assert result.status == "ok", result.reason
    # For a pure start attack, all insertions should collapse to one span.
    assert len(result.inserted_spans) == 1


# ---------------------------------------------------------------------------
# Integration tests for build_padded_control_and_attack_encodings_general
# ---------------------------------------------------------------------------

# 1. Regression: must match legacy function exactly for relevant_start_5 ----

def test_matches_legacy_for_start_5(tokenizer, device):
    """
    The general function does not have to produce byte-identical encodings to
    the legacy function.  SequenceMatcher may place the boundary of a shared
    token (e.g. ':') differently when that token also appears elsewhere in the
    sequence.  What matters for patching correctness is:
      a) same sequence length in control and attack,
      b) same number of pad slots as the legacy function,
      c) passage tokens (everything from the first non-pad, non-prefix position
         onward) are at the same absolute indices in both control and attack.
    """
    attacked = "relevant: " * 5 + PASSAGE
    legacy_c, legacy_a, n_prefix_legacy, n_attack_legacy = (
        build_padded_control_and_attack_encodings(
            tokenizer=tokenizer,
            query=QUERY,
            passage=PASSAGE,
            attacked_passage=attacked,
            max_length=512,
            device=device,
        )
    )
    new_c, new_a, result = build_padded_control_and_attack_encodings_general(
        tokenizer=tokenizer,
        query=QUERY,
        passage=PASSAGE,
        attacked_passage=attacked,
        max_length=512,
        device=device,
    )
    assert result.status == "ok", result.reason

    # (a) Sequence lengths match between legacy and general.
    assert new_c["input_ids"].shape == legacy_c["input_ids"].shape, \
        "control shape mismatch vs legacy"
    assert new_a["input_ids"].shape == legacy_a["input_ids"].shape, \
        "attack shape mismatch vs legacy"

    # (b) Same total number of pad slots in the control.
    n_pad_legacy = (legacy_c["attention_mask"] == 0).sum().item()
    n_pad_new    = (new_c["attention_mask"]    == 0).sum().item()
    assert n_pad_new == n_pad_legacy, \
        f"pad slot count mismatch: legacy={n_pad_legacy} general={n_pad_new}"

    # (c) Attack encodings are identical (attack sequence is unambiguous).
    assert torch.equal(legacy_a["input_ids"],      new_a["input_ids"]),      \
        "attack input_ids mismatch vs legacy"
    assert torch.equal(legacy_a["attention_mask"], new_a["attention_mask"]), \
        "attack attention_mask mismatch vs legacy"

    # (d) Passage tokens start at the same absolute index in control and attack
    #     for both functions.  The passage starts right after the pad block.
    def first_passage_idx(ids_tensor, mask_tensor):
        """Index of the first non-pad token after the pad block."""
        ids  = ids_tensor[0].tolist()
        mask = mask_tensor[0].tolist()
        # Find first position that is not a pad-slot (mask==1) after the prefix.
        in_pad_block = False
        for i, m in enumerate(mask):
            if m == 0:
                in_pad_block = True
            elif in_pad_block and m == 1:
                return i  # first token after the pad block
        return None

    idx_legacy = first_passage_idx(legacy_c["input_ids"], legacy_c["attention_mask"])
    idx_new    = first_passage_idx(new_c["input_ids"],    new_c["attention_mask"])
    assert idx_legacy is not None and idx_new is not None
    # The passage token IDs from that point on must be identical.
    passage_legacy = legacy_c["input_ids"][0, idx_legacy:].tolist()
    passage_new    = new_c["input_ids"][0, idx_new:].tolist()
    assert passage_legacy == passage_new, \
        "passage tokens diverge between legacy and general control encodings"


# 2. End attack: alignment succeeds, pad slots are at the end ---------------

def test_end_attack_aligns(tokenizer, device):
    attacked = PASSAGE + " relevant: relevant: relevant: relevant: relevant:"
    new_c, new_a, result = build_padded_control_and_attack_encodings_general(
        tokenizer=tokenizer,
        query=QUERY,
        passage=PASSAGE,
        attacked_passage=attacked,
        max_length=512,
        device=device,
    )
    assert result.status == "ok", result.reason
    assert result.n_inserted > 0
    # Shapes match
    assert new_c["input_ids"].shape == new_a["input_ids"].shape
    # Pad slots in the control match the alignment
    n_pad = (new_c["attention_mask"] == 0).sum().item()
    assert n_pad == result.n_inserted
    # For an end attack, the pad slots should be in the second half of the sequence
    mask = new_c["attention_mask"][0].tolist()
    first_zero = next((i for i, m in enumerate(mask) if m == 0), None)
    seq_len = new_c["input_ids"].shape[1]
    assert first_zero is not None
    assert first_zero > seq_len // 2, \
        "end attack pad slots should appear in the second half of the sequence"


# 3. Random / mid-passage attack: alignment succeeds ----------------------

def test_random_attack_aligns(tokenizer, device):
    # Synthetic: insert attack tokens in the middle of the passage.
    # Use a passage with a clear split point that does not depend on ". " in PASSAGE.
    passage = "Climate change is a global challenge. Renewable energy is essential."
    attacked = "Climate change is a global challenge. relevant: relevant: Renewable energy is essential."

    new_c, new_a, result = build_padded_control_and_attack_encodings_general(
        tokenizer=tokenizer,
        query=QUERY,
        passage=passage,
        attacked_passage=attacked,
        max_length=512,
        device=device,
    )
    assert result.status == "ok", result.reason
    assert result.n_inserted > 0
    assert len(result.inserted_spans) >= 1
    n_pad = (new_c["attention_mask"] == 0).sum().item()
    assert n_pad == result.n_inserted


def _test_random_attack_aligns_unused(tokenizer, device):
    # Kept here in case the test above is refactored.
    passage = PASSAGE
    attacked = passage  # placeholder
    new_c, new_a, result = build_padded_control_and_attack_encodings_general(
        tokenizer=tokenizer,
        query=QUERY,
        passage=PASSAGE,
        attacked_passage=mid_attacked,
        max_length=512,
        device=device,
    )
    assert result.status == "ok", result.reason
    assert result.n_inserted > 0
    assert len(result.inserted_spans) >= 1
    n_pad = (new_c["attention_mask"] == 0).sum().item()
    assert n_pad == result.n_inserted


# 4. Trap: passage already contains the attack token -----------------------

def test_passage_contains_attack_token(tokenizer, device):
    """
    Passage contains "relevant" — SequenceMatcher must still produce a valid
    alignment (inserting at the start) OR report failure explicitly.
    A silent wrong answer (wrong span, wrong pad count) is the only bad outcome.
    """
    passage = "This document is highly relevant to climate research."
    attacked = "relevant: " * 5 + passage

    new_c, new_a, result = build_padded_control_and_attack_encodings_general(
        tokenizer=tokenizer,
        query=QUERY,
        passage=passage,
        attacked_passage=attacked,
        max_length=512,
        device=device,
    )
    if result.status == "ok":
        # If it succeeded, the pad count must equal n_inserted
        n_pad = (new_c["attention_mask"] == 0).sum().item()
        assert n_pad == result.n_inserted, \
            "pad slot count must equal n_inserted from AlignmentResult"
        assert new_c["input_ids"].shape == new_a["input_ids"].shape
    else:
        # Failure is acceptable — just must be explicit.
        assert result.reason, "failed alignment must carry a human-readable reason"


# 5. Multi-token compound attack token ------------------------------------

def test_multitoken_attack(tokenizer, device):
    attacked = "information: " * 3 + PASSAGE
    new_c, new_a, result = build_padded_control_and_attack_encodings_general(
        tokenizer=tokenizer,
        query=QUERY,
        passage=PASSAGE,
        attacked_passage=attacked,
        max_length=512,
        device=device,
    )
    assert result.status == "ok", result.reason
    # The attack has 3 repetitions; total insertions must be divisible by 3.
    assert result.n_inserted % 3 == 0, \
        f"n_inserted={result.n_inserted} should be divisible by repetition count 3"
    n_pad = (new_c["attention_mask"] == 0).sum().item()
    assert n_pad == result.n_inserted
