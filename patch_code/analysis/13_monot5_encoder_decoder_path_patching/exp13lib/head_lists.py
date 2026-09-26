"""
exp13lib/head_lists.py
=======================
Loads the two fixed head sets for Experiment 13 and asserts their sizes.

Encoder senders (18 heads)
---------------------------
configs/heads/encoder_senders.json — the encoder self-attention head-slots
whose mean combined effect exceeded 0.02 on Experiment 11's CANONICAL run
(single attack `relevant_start_5`, n=100) — see Section 4.5 of the report
and exp11lib/... outputs/head_summary_canonical.csv. This is the file the
report's "18 of 144 head-slots" sentence describes (verified by
reproducing the exact table: L10-H0=0.182, L10-H2=0.161, ..., L10-H7=0.027).

Decoder receivers (31 heads)
-------------------------------
configs/heads/decoder_receivers.json — the top 31 decoder_cross_attn
head-slots by mean combined effect on Experiment 3's Grid A (105 attacks,
n=100, outputs/grid_a/aggregated/head_summary.csv).

IMPORTANT PROVENANCE NOTE (read before changing this file)
-------------------------------------------------------------
The Experiment 3 report text claims "33 of 288 head-slots exceed 0.02"
(31 cross-attn + 2 self-attn: L11-S-H3, L10-S-H3). That claim does NOT
reproduce from any on-disk aggregate we could find:
  - flat grid_a/aggregated/head_summary.csv (n=10432): 22 heads > 0.02
    (21 decoder_cross_attn + 1 decoder_self_attn), and L10-S-H3 is
    actually -0.031 there, not +0.021 as the report states.
  - attack-balanced grid_a (mean of 105 per-attack means): 21 heads > 0.02.
  - grid_b (canonical single attack): 11 heads > 0.02, all cross-attn.
The 22-head flat-threshold set is *exactly* Experiment 2's
`flagged_heads.json` ("22-head steering subset"), which this experiment
was explicitly told NOT to reuse.

Per user decision (2026-08-09): since no on-disk aggregate reproduces the
report's 33/31, the 31 decoder_cross_attn receivers here are instead the
TOP 31 decoder_cross_attn head-slots by combined_effect_mean in
grid_a/aggregated/head_summary.csv (flat, n=100), taken by RANK rather
than by the (unreproducible) 0.02 threshold. Excludes decoder_self_attn
entirely, per the task's explicit instruction. This is fully reproducible
from data already checked into the repo; see
scripts/00_derive_head_lists.py for the exact derivation.
"""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass
from typing import List

HEADS_DIR = pathlib.Path(__file__).parent.parent / "configs" / "heads"

N_SENDERS_EXPECTED = 18
N_RECEIVERS_EXPECTED = 31
N_PATHS_EXPECTED = N_SENDERS_EXPECTED * N_RECEIVERS_EXPECTED  # 558


@dataclass(frozen=True)
class SenderHead:
    layer: int
    head_idx: int
    label: str  # e.g. "L10H0"


@dataclass(frozen=True)
class ReceiverHead:
    layer: int
    head_idx: int
    component: str  # always "decoder_cross_attn"
    label: str  # e.g. "L11-X-H3"


def load_senders(path: pathlib.Path = None) -> List[SenderHead]:
    path = path or (HEADS_DIR / "encoder_senders.json")
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)
    senders = [SenderHead(layer=r["layer"], head_idx=r["head_idx"], label=r["label"]) for r in raw]
    if len(senders) != N_SENDERS_EXPECTED:
        raise ValueError(f"Expected {N_SENDERS_EXPECTED} encoder senders, got {len(senders)} from {path}")
    if len({(s.layer, s.head_idx) for s in senders}) != len(senders):
        raise ValueError(f"Duplicate sender head entries in {path}")
    return senders


def load_receivers(path: pathlib.Path = None) -> List[ReceiverHead]:
    path = path or (HEADS_DIR / "decoder_receivers.json")
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)
    receivers = [
        ReceiverHead(layer=r["layer"], head_idx=r["head_idx"], component=r["component"], label=r["label"])
        for r in raw
    ]
    if len(receivers) != N_RECEIVERS_EXPECTED:
        raise ValueError(f"Expected {N_RECEIVERS_EXPECTED} decoder receivers, got {len(receivers)} from {path}")
    if any(r.component != "decoder_cross_attn" for r in receivers):
        raise ValueError(f"Non-cross-attention receiver found in {path} — self-attention heads must be excluded.")
    if len({(r.layer, r.head_idx) for r in receivers}) != len(receivers):
        raise ValueError(f"Duplicate receiver head entries in {path}")
    return receivers


def receiver_layers(receivers: List[ReceiverHead]) -> List[int]:
    """Distinct decoder layers that host at least one receiver head, sorted."""
    return sorted({r.layer for r in receivers})


def sender_layers(senders: List[SenderHead]) -> List[int]:
    """Distinct encoder layers that host at least one sender head, sorted."""
    return sorted({s.layer for s in senders})


def assert_head_sets(senders: List[SenderHead], receivers: List[ReceiverHead]) -> None:
    """Sanity check #5 from the task spec: correct head-set sizes."""
    assert len(senders) == N_SENDERS_EXPECTED, f"{len(senders)} != {N_SENDERS_EXPECTED}"
    assert len(receivers) == N_RECEIVERS_EXPECTED, f"{len(receivers)} != {N_RECEIVERS_EXPECTED}"
    assert len(senders) * len(receivers) == N_PATHS_EXPECTED
