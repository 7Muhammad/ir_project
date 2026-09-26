"""Shared fixtures: tiny random T5 (CPU, no download) + real monoT5 tokenizer."""

from __future__ import annotations

import pathlib
import sys

import pytest
import torch

EXP_DIR = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EXP_DIR))
import exp15lib  # noqa: E402,F401  (sibling sys.path setup)

from transformers import T5Config, T5ForConditionalGeneration  # noqa: E402

TRUE_ID, FALSE_ID, PAD_ID = 5, 7, 0
T = 16
Q_POS = [3, 4, 5, 6]          # "query text" positions in the synthetic prompt
A_POS = [9, 10, 13]           # scattered "attack" positions (two spans)
EXP1_ATTACKS = EXP_DIR.parent / "01_monot5_layer_patching" / "outputs" / "attacks"


@pytest.fixture(scope="session")
def tiny_model():
    torch.manual_seed(0)
    cfg = T5Config(vocab_size=100, d_model=32, d_kv=8, num_heads=4, d_ff=64, num_layers=3,
                   num_decoder_layers=2, decoder_start_token_id=0, dropout_rate=0.0)
    m = T5ForConditionalGeneration(cfg)
    m.eval()
    return m


@pytest.fixture(scope="session")
def encs():
    """Attack + matched padded control (same length; A -> pad with mask 0)."""
    g = torch.Generator().manual_seed(3)
    ids = torch.randint(2, 100, (1, T), generator=g)
    atk = {"input_ids": ids, "attention_mask": torch.ones(1, T, dtype=torch.long)}
    c_ids = ids.clone()
    c_mask = torch.ones(1, T, dtype=torch.long)
    c_ids[0, A_POS] = PAD_ID
    c_mask[0, A_POS] = 0
    return {"control": {"input_ids": c_ids, "attention_mask": c_mask}, "attack": atk}


@pytest.fixture(scope="session")
def tol():
    return {"decomposition_atol": 1e-5, "control_message_max_norm": 1e-6}


@pytest.fixture(scope="session")
def tokenizer():
    from transformers import T5Tokenizer
    return T5Tokenizer.from_pretrained("castorini/monot5-base-msmarco", use_fast=False)
