"""
exp19lib — Experiment 19: head-level follow-up to Exp 18 (encoder representation change, control -> attack).

Representation: the "pre-o_proj head-output representation" H[l, h, t] in R^64 — the input of
encoder.block[l].layer[0].SelfAttention.o sliced [h*64, (h+1)*64) — i.e. exactly the tensor read by
exp16lib.heads.EncoderHeadCapture (Exp 16 stages 09/13/18, stage-22 token sample). NOT the full encoder state.

Earlier experiments are imported, never copied:
  Exp 18  exp18lib.repr_change   regions (pair_regions / check_pair / collate_pairs / check_batch), 1 - cos,
                                 normalized L2, linear CKA (direct + streaming), groups, query bootstrap
  Exp 18  outputs/01_repr_change per-instance metadata + scores (population source of truth; read-only)
  Exp 16  exp16lib.*             encodings, paired-manifest constants, model loading, EncoderHeadCapture
  Exp 16  outputs/17_paired_manifest   attacked texts (read-only)
  Exp 13  exp13lib.head_lists.load_senders   the 18 causally important encoder heads (Exp 11 canonical, > 0.02)
  Exp 11  outputs/head_summary_canonical.csv per-head combined causal effect (continuous; read-only)
"""

from __future__ import annotations

import pathlib
import sys

EXP_DIR = pathlib.Path(__file__).resolve().parent.parent
ANALYSIS_DIR = EXP_DIR.parent
EXP18_DIR = ANALYSIS_DIR / "18_monot5_encoder_representation_change"
EXP16_DIR = ANALYSIS_DIR / "16_monot5_artificial_query_document_similarity"
EXP11_DIR = ANALYSIS_DIR / "11_monot5_encoder_head_patching"
for _p in (EXP18_DIR, EXP_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import exp18lib  # noqa: E402,F401  (puts Exp 16 and its sibling paths on sys.path)
