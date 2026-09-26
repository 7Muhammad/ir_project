"""Path setup + shared fixtures: a tiny randomly-initialised T5 (CPU, no downloads)."""

from __future__ import annotations

import pathlib
import sys

import pytest
import torch

TESTS_DIR = pathlib.Path(__file__).parent.resolve()
EXP_DIR = TESTS_DIR.parent                              # 03_monot5_head_patching_ablation
EXP1_DIR = EXP_DIR.parent / "01_monot5_layer_patching"  # sibling Experiment 1
sys.path.insert(0, str(EXP1_DIR))                # src.*  (Experiment 1)
sys.path.insert(0, str(EXP_DIR))                 # headlib.*

from transformers import T5Config, T5ForConditionalGeneration  # noqa: E402

TRUE_ID, FALSE_ID = 5, 7
DEVICE = torch.device("cpu")


@pytest.fixture(scope="session")
def tiny_model():
    torch.manual_seed(0)
    cfg = T5Config(
        vocab_size=100,
        d_model=32,
        d_kv=8,
        num_heads=4,
        d_ff=64,
        num_layers=2,
        num_decoder_layers=2,
        decoder_start_token_id=0,
        dropout_rate=0.0,
    )
    model = T5ForConditionalGeneration(cfg)
    model.eval()
    return model


def _random_encoding(seed: int, length: int) -> dict:
    g = torch.Generator().manual_seed(seed)
    return {
        "input_ids": torch.randint(2, 100, (1, length), generator=g),
        "attention_mask": torch.ones(1, length, dtype=torch.long),
    }


@pytest.fixture(scope="session")
def control_enc():
    return _random_encoding(seed=1, length=11)


@pytest.fixture(scope="session")
def attack_enc():
    # Same length as control (as guaranteed by the padded-control construction).
    return _random_encoding(seed=2, length=11)


@pytest.fixture(scope="session")
def clean_enc():
    # Different length — decoder head activations are (1, 1, inner) regardless.
    return _random_encoding(seed=3, length=7)
