"""Path setup + shared fixtures for Experiment 10 tests."""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pytest
import torch

TESTS_DIR = pathlib.Path(__file__).parent.resolve()
EXP_DIR = TESTS_DIR.parent                              # 10_monot5_encoder_diff_geometry
EXP1_DIR = EXP_DIR.parent / "01_monot5_layer_patching"  # sibling Experiment 1
EXP6_DIR = EXP_DIR.parent / "06_monot5_query_doc_patching"  # sibling Experiment 6
sys.path.insert(0, str(EXP1_DIR))                       # src.*
sys.path.insert(0, str(EXP6_DIR))                       # exp6lib.*
sys.path.insert(0, str(EXP_DIR))                        # exp10lib.*

from transformers import T5Config, T5ForConditionalGeneration, T5Tokenizer  # noqa: E402

DEVICE = torch.device("cpu")
REAL_CHECKPOINT = "castorini/monot5-base-msmarco"


@pytest.fixture(scope="session")
def real_tokenizer():
    return T5Tokenizer.from_pretrained(REAL_CHECKPOINT, use_fast=False)


@pytest.fixture(scope="session")
def real_example():
    """One real (query, passage, attacked_passage) example reused from Experiment 1's outputs."""
    path = EXP1_DIR / "outputs" / "attacks" / "relevant_start_5" / "scores" / "selected_examples.jsonl"
    with open(path, encoding="utf-8") as fh:
        return json.loads(fh.readline())


@pytest.fixture(scope="session")
def tiny_model():
    torch.manual_seed(0)
    cfg = T5Config(
        vocab_size=100, d_model=32, d_kv=8, num_heads=4, d_ff=64,
        num_layers=12, num_decoder_layers=1, decoder_start_token_id=0,
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
    return _random_encoding(seed=1, length=14)


@pytest.fixture(scope="session")
def attack_enc():
    return _random_encoding(seed=2, length=14)  # same length, as guaranteed by padded-control construction


def make_low_rank_diffs(n: int, d: int, direction_seed: int = 0, noise_scale: float = 0.05, n_directions: int = 1) -> np.ndarray:
    """
    Synthetic diff matrix with a known rank structure, for testing geometry.py.
    n_directions=1 -> near rank-1 (dominant direction + small isotropic noise).
    """
    rng = np.random.default_rng(direction_seed)
    directions = rng.normal(size=(n_directions, d))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    coeffs = rng.normal(loc=1.0, scale=0.1, size=(n, n_directions))
    signal = coeffs @ directions
    noise = rng.normal(scale=noise_scale, size=(n, d))
    return signal + noise
