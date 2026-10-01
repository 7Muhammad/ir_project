"""Shared fixtures: real monoT5 tokenizer (cached, no download)."""

from __future__ import annotations

import pathlib
import sys

import pytest

EXP_DIR = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EXP_DIR))
import exp18lib  # noqa: E402,F401


@pytest.fixture(scope="session")
def tokenizer():
    from transformers import T5Tokenizer
    return T5Tokenizer.from_pretrained("castorini/monot5-base-msmarco", use_fast=False)
