"""
exp18lib — Experiment 18: full-population encoder REPRESENTATION CHANGE, padded control -> attacked input.

For every aligned judged DL19 instance (qrel 2/3 or 0) x 105 attacks, compare the full encoder residual
stream (768-d) of the padded control and the attacked input per encoder layer, for the query, the original
document and the whole sequence (shared tokens / full visible). Metrics are reduced on the fly; no hidden
state is stored.

Earlier experiments are imported, never copied (project reuse convention):
  Exp 16  exp16lib.*   encodings (inputs.encode_attack_and_control), paired-manifest constants, model loading,
                       status helpers
  Exp 16  outputs/     17_paired_manifest (population, alignment failures) and 18_paired_forward (scores,
                       success labels) — read-only
"""

from __future__ import annotations

import pathlib
import sys

EXP_DIR = pathlib.Path(__file__).resolve().parent.parent
ANALYSIS_DIR = EXP_DIR.parent
EXP16_DIR = ANALYSIS_DIR / "16_monot5_artificial_query_document_similarity"
for _p in (EXP16_DIR, EXP_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import exp16lib  # noqa: E402,F401  (sets up the Exp 01/03/06/13/14 sibling paths)
