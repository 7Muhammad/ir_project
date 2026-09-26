"""Path setup for Experiment 9's tests. Most of exp9lib is pure-Python
(no model needed); tests that touch scoring monkeypatch it instead of
loading a real tokenizer, since a tiny random T5Config has no matching
tokenizer vocabulary."""

from __future__ import annotations

import pathlib
import sys

TESTS_DIR = pathlib.Path(__file__).parent.resolve()
EXP_DIR = TESTS_DIR.parent                                       # 09_monot5_score_vs_rank
EXP1_DIR = EXP_DIR.parent / "01_monot5_layer_patching"           # sibling Experiment 1
EXP3_DIR = EXP_DIR.parent / "03_monot5_head_patching_ablation"   # sibling Experiment 3
DECODERLENS_DIR = EXP_DIR.parent / "monot5_decoderlens_rank"     # sibling DecoderLens
sys.path.insert(0, str(EXP1_DIR))                # src.*  (Experiment 1)
sys.path.insert(0, str(EXP3_DIR))                # headlib.*
sys.path.insert(0, str(EXP_DIR))                 # exp9lib.*
