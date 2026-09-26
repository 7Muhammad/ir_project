"""Path setup + shared fixtures. query_words/structural_masks need the REAL
monoT5 tokenizer (SentencePiece boundaries are checkpoint-specific), but no
model forward pass -- these tests are CPU-only and fast."""

from __future__ import annotations

import pathlib
import sys

import pytest

TESTS_DIR = pathlib.Path(__file__).parent.resolve()
EXP_DIR = TESTS_DIR.parent                               # 12_query_token_localization
EXP1_DIR = EXP_DIR.parent / "01_monot5_layer_patching"
EXP6_DIR = EXP_DIR.parent / "06_monot5_query_doc_patching"
sys.path.insert(0, str(EXP1_DIR))   # src.*
sys.path.insert(0, str(EXP6_DIR))   # exp6lib.*
sys.path.insert(0, str(EXP_DIR))    # exp12lib.*

from transformers import T5Tokenizer  # noqa: E402


@pytest.fixture(scope="session")
def tokenizer():
    return T5Tokenizer.from_pretrained("castorini/monot5-base-msmarco", use_fast=False)
