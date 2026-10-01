"""Shared fixtures: tiny random T5 (CPU), real monoT5 tokenizer."""

from __future__ import annotations

import pathlib
import sys

import pytest
import torch

EXP_DIR = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EXP_DIR))
import exp19lib  # noqa: E402,F401

from transformers import T5Config, T5ForConditionalGeneration  # noqa: E402


@pytest.fixture(scope="session")
def tiny_model():
    torch.manual_seed(0)
    cfg = T5Config(vocab_size=100, d_model=32, d_kv=8, num_heads=4, d_ff=64, num_layers=3,
                   num_decoder_layers=2, decoder_start_token_id=0, dropout_rate=0.0)
    return T5ForConditionalGeneration(cfg).eval()


@pytest.fixture(scope="session")
def tokenizer():
    from transformers import T5Tokenizer
    return T5Tokenizer.from_pretrained("castorini/monot5-base-msmarco", use_fast=False)
