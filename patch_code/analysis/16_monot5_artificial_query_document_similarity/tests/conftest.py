"""Shared fixtures: tiny random T5 (CPU, no download), real monoT5 tokenizer, script-module loader."""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest
import torch

EXP_DIR = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EXP_DIR))
import exp16lib  # noqa: E402,F401

from transformers import T5Config, T5ForConditionalGeneration  # noqa: E402

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
def tokenizer():
    from transformers import T5Tokenizer
    return T5Tokenizer.from_pretrained("castorini/monot5-base-msmarco", use_fast=False)


def load_script(name: str):
    """Import scripts/<name>.py as a module (scripts start with digits)."""
    path = EXP_DIR / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"exp16_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def exp1_pair(attack: str, idx: int = 0):
    import json
    with open(EXP1_ATTACKS / attack / "pairs" / "pairs.jsonl", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if i == idx:
                return json.loads(line)
    raise IndexError(idx)
