"""
exp14lib/head_lists.py
========================
Candidate single-head list for Experiment 14, Phase 1.

Reuses (does not re-derive) Experiment 13's already-vetted, checked-in head
lists:
  - configs/heads/encoder_senders.json    (18 encoder_self_attn heads,
    combined_effect_mean > 0.02 on Experiment 11's canonical aggregate)
  - configs/heads/decoder_receivers.json  (31 decoder_cross_attn heads,
    top-31 by combined_effect_mean rank on Experiment 3's grid_a aggregate)

IMPORTANT PROVENANCE NOTE (read before changing this file) — see
DECISIONS.md item 1 for the full writeup. current_report.tex's abstract and
Experiment 3's own report claim "33 of 288 decoder head-slots (31
cross-attention + 2 self-attention: L11-S-H3, L10-S-H3)". That count does
NOT reproduce from any on-disk aggregate (verified independently here and
already documented in Experiment 13's exp13lib/head_lists.py): a flat 0.02
threshold on grid_a/aggregated/head_summary.csv gives 22 heads (21
cross-attn + 1 self-attn), not 33/31. Per the SAME prior user decision
Experiment 13 recorded (2026-08-09), Experiment 14 uses the reproducible,
rank-based 31-head decoder_cross_attn set and excludes decoder_self_attn
entirely — 18 encoder + 31 decoder = 49 candidate single-head interventions,
not "18 + 33".

Encoder heads additionally fan out into 5 position-mask conditions each
(document / query / template / query_document / all_valid — see
exp14lib/position_masks.py), so the full Phase-1 candidate-intervention
count is 18*5 + 31 = 121 (head, condition) pairs. Decoder heads have exactly
one condition ("decoder") since monoT5 scoring uses a single decoder step
with no sequence-position choice to make (see Experiment 2's identical
argument in exp2lib/intervene.py).
"""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass, asdict
from typing import List

N_ENCODER_HEADS_EXPECTED = 18
N_DECODER_HEADS_EXPECTED = 31

ENCODER_POSITION_MASKS: List[str] = ["document", "query", "template", "query_document", "all_valid"]


@dataclass(frozen=True)
class EncoderHead:
    layer: int
    head_idx: int
    label: str
    combined_effect_mean: float


@dataclass(frozen=True)
class DecoderHead:
    layer: int
    head_idx: int
    component: str  # always "decoder_cross_attn"
    label: str
    combined_effect_mean: float


def derive_candidate_heads(exp13_heads_dir: pathlib.Path) -> dict:
    """
    Load Experiment 13's encoder_senders.json / decoder_receivers.json and
    assemble the Experiment-14 candidate-head manifest. Fails loudly (raises)
    if the counts don't match what's documented above rather than silently
    accepting a different set.
    """
    with open(exp13_heads_dir / "encoder_senders.json", encoding="utf-8") as fh:
        raw_enc = json.load(fh)
    with open(exp13_heads_dir / "decoder_receivers.json", encoding="utf-8") as fh:
        raw_dec = json.load(fh)

    encoder_heads = [
        EncoderHead(
            layer=r["layer"], head_idx=r["head_idx"], label=r["label"],
            combined_effect_mean=r["combined_effect_mean_canonical"],
        )
        for r in raw_enc
    ]
    decoder_heads = [
        DecoderHead(
            layer=r["layer"], head_idx=r["head_idx"], component=r["component"], label=r["label"],
            combined_effect_mean=r["combined_effect_mean_grid_a_n100"],
        )
        for r in raw_dec
    ]

    if len(encoder_heads) != N_ENCODER_HEADS_EXPECTED:
        raise ValueError(
            f"Expected {N_ENCODER_HEADS_EXPECTED} encoder heads from "
            f"{exp13_heads_dir / 'encoder_senders.json'}, got {len(encoder_heads)}. "
            "Experiment 13's head list may have changed — investigate before proceeding "
            "(see this module's docstring for the expected provenance)."
        )
    if len(decoder_heads) != N_DECODER_HEADS_EXPECTED:
        raise ValueError(
            f"Expected {N_DECODER_HEADS_EXPECTED} decoder heads from "
            f"{exp13_heads_dir / 'decoder_receivers.json'}, got {len(decoder_heads)}. "
            "Experiment 13's head list may have changed — investigate before proceeding."
        )
    if any(h.component != "decoder_cross_attn" for h in decoder_heads):
        raise ValueError("Non-cross-attention decoder head found — self-attention must be excluded (see DECISIONS.md).")

    n_encoder_candidates = len(encoder_heads) * len(ENCODER_POSITION_MASKS)
    n_decoder_candidates = len(decoder_heads)

    return {
        "provenance": {
            "encoder_source": str((exp13_heads_dir / "encoder_senders.json").resolve()),
            "decoder_source": str((exp13_heads_dir / "decoder_receivers.json").resolve()),
            "note": (
                "current_report.tex's '33 of 288 (31 cross-attn + 2 self-attn)' does not "
                "reproduce from any on-disk aggregate; this manifest uses the reproducible "
                "18 encoder (threshold 0.02) + 31 decoder cross-attn (rank top-31) sets "
                "Experiment 13 already settled on. See DECISIONS.md item 1."
            ),
        },
        "encoder_position_masks": ENCODER_POSITION_MASKS,
        "encoder_heads": [asdict(h) for h in encoder_heads],
        "decoder_heads": [asdict(h) for h in decoder_heads],
        "n_encoder_heads": len(encoder_heads),
        "n_decoder_heads": len(decoder_heads),
        "n_encoder_candidates": n_encoder_candidates,
        "n_decoder_candidates": n_decoder_candidates,
        "n_total_candidates": n_encoder_candidates + n_decoder_candidates,
    }


def load_candidate_heads(path: pathlib.Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    if manifest["n_encoder_heads"] != N_ENCODER_HEADS_EXPECTED:
        raise ValueError(f"candidate_heads.json has {manifest['n_encoder_heads']} encoder heads, expected {N_ENCODER_HEADS_EXPECTED}")
    if manifest["n_decoder_heads"] != N_DECODER_HEADS_EXPECTED:
        raise ValueError(f"candidate_heads.json has {manifest['n_decoder_heads']} decoder heads, expected {N_DECODER_HEADS_EXPECTED}")
    return manifest


def encoder_heads_from_manifest(manifest: dict) -> List[EncoderHead]:
    return [EncoderHead(**r) for r in manifest["encoder_heads"]]


def decoder_heads_from_manifest(manifest: dict) -> List[DecoderHead]:
    return [DecoderHead(**r) for r in manifest["decoder_heads"]]


def apply_head_caps(manifest: dict, max_encoder_heads: int = None, max_decoder_heads: int = None) -> dict:
    """
    Truncate the candidate-head manifest for smoke-test runs. Keeps the
    highest-combined_effect heads first (manifest lists are already sorted
    that way by Experiment 13's derivation script). NEVER used to change
    the canonical 18/31 counts written by 00_derive_candidate_heads.py —
    only downstream scripts apply this, and only when configured to.
    """
    capped = dict(manifest)
    if max_encoder_heads is not None:
        capped["encoder_heads"] = manifest["encoder_heads"][:max_encoder_heads]
    if max_decoder_heads is not None:
        capped["decoder_heads"] = manifest["decoder_heads"][:max_decoder_heads]
    capped["n_encoder_heads"] = len(capped["encoder_heads"])
    capped["n_decoder_heads"] = len(capped["decoder_heads"])
    capped["n_encoder_candidates"] = capped["n_encoder_heads"] * len(manifest["encoder_position_masks"])
    capped["n_decoder_candidates"] = capped["n_decoder_heads"]
    capped["n_total_candidates"] = capped["n_encoder_candidates"] + capped["n_decoder_candidates"]
    return capped
