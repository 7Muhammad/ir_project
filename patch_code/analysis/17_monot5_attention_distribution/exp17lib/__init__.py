"""
exp17lib — Experiment 17: encoder attention-distribution statistics (shape, not raw mass).

Exploratory, observational screen on EXACTLY the Exp 16 stage-22 token sample (253 base pairs,
6,325 clean / padded-control / attacked sequences). The stage-22 tokenised inputs are re-run
through the monoT5 encoder with attention outputs enabled; per-row attention statistics
(entropy, concentration, effective support, locality, injected-token mass) are reduced to
scalars on the fly and the attention matrices are discarded (never stored).

Earlier experiments are imported, never copied (project reuse convention):
  Exp 16  exp16lib.*   stage-22 loader (TokenSample), region masks, model loading, canonical
                       encoder head list (Exp 13 senders), Cohen d / paired dz / AUROC helpers,
                       plot palette (scripts/05_plot.py, scripts/21_plot_paired.py)
  Exp 11  outputs/     per-head causal combined effects (read-only; optional comparison)
"""

from __future__ import annotations

import pathlib
import sys

EXP_DIR = pathlib.Path(__file__).resolve().parent.parent
ANALYSIS_DIR = EXP_DIR.parent
EXP16_DIR = ANALYSIS_DIR / "16_monot5_artificial_query_document_similarity"
EXP11_DIR = ANALYSIS_DIR / "11_monot5_encoder_head_patching"
for _p in (EXP16_DIR, EXP_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import exp16lib  # noqa: E402,F401  (sets up the Exp 01/03/06/13/14 sibling paths)
