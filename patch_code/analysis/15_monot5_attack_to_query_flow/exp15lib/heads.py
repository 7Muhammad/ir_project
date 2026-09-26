"""
exp15lib/heads.py
==================
The canonical 18 important encoder heads, loaded with Experiment 13's own
loader (`exp13lib.head_lists.load_senders`, which asserts exactly 18 unique
(layer, head) entries). Order is preserved from the JSON (descending Exp 11
canonical combined effect) and used as the display order everywhere.

Only smoke.yaml may restrict to a subset (`heads.subset`); the full config
must run all 18 (DECISIONS.md items 1-4).
"""

from __future__ import annotations

from typing import List

import exp15lib  # noqa: F401
from exp13lib.head_lists import SenderHead, load_senders

from exp15lib.config import resolve_cfg_path


def load_heads(cfg: dict) -> List[SenderHead]:
    heads = load_senders(resolve_cfg_path(cfg, cfg["heads"]["encoder_heads_json"]))
    subset = cfg["heads"].get("subset")
    if subset:
        by_label = {h.label: h for h in heads}
        missing = [s for s in subset if s not in by_label]
        if missing:
            raise ValueError(f"heads.subset labels not among the 18 canonical heads: {missing}")
        heads = [h for h in heads if h.label in set(subset)]
    return heads


def heads_by_label(cfg: dict, labels: List[str]) -> List[SenderHead]:
    all_heads = {h.label: h for h in load_senders(resolve_cfg_path(cfg, cfg["heads"]["encoder_heads_json"]))}
    missing = [l for l in labels if l not in all_heads]
    if missing:
        raise ValueError(f"Unknown head labels: {missing}")
    return [all_heads[l] for l in labels]
