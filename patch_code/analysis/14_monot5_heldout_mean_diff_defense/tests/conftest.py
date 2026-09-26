from __future__ import annotations

import pathlib
import sys

import pytest

TESTS_DIR = pathlib.Path(__file__).parent.resolve()
EXP_DIR = TESTS_DIR.parent
EXP1_DIR = EXP_DIR.parent / "01_monot5_layer_patching"
EXP2_DIR = EXP_DIR.parent / "02_monot5_mean_diff_intervention"
EXP3_DIR = EXP_DIR.parent / "03_monot5_head_patching_ablation"
EXP6_DIR = EXP_DIR.parent / "06_monot5_query_doc_patching"
for d in (EXP1_DIR, EXP2_DIR, EXP3_DIR, EXP6_DIR, EXP_DIR):
    sys.path.insert(0, str(d))


@pytest.fixture(scope="session")
def tokenizer():
    """
    Real monoT5 tokenizer (lightweight vs. the full model — just SentencePiece
    vocab) — needed for the position-mask tests, which must exercise the
    ACTUAL tokenization boundaries, not a synthetic stand-in.
    """
    from transformers import T5Tokenizer
    return T5Tokenizer.from_pretrained("castorini/monot5-base-msmarco", use_fast=False)
