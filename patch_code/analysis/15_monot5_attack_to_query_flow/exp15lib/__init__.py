"""
exp15lib — Experiment 15: attack-token -> query information flow.

Source-specific edge-message patching at the 18 canonical encoder heads:
replace ONLY the part of a head's pre-`.o` output at query positions that
originates from the injected attack-token positions,

    m(q) = sum_{a in A} P[q, a] V[a],

between the attacked run and the matched padded-control run (both
directions), and measure the change in the monoT5 score.

Earlier experiments are imported, never copied (project reuse convention):
  Exp 01  src.*        model loading, padded-control construction, alignment,
                       SKIP_EPSILON
  Exp 03  headlib.*    config loading with attacks.inherit_from
  Exp 06  exp6lib.*    query/document span detection
  Exp 11  exp11lib.*   encoder head geometry, .o hooks, whole-head patching
  Exp 13  exp13lib.*   canonical 18-head encoder list loader
  Exp 14  exp14lib.*   status.json resume helpers
"""

from __future__ import annotations

import pathlib
import sys

EXP_DIR = pathlib.Path(__file__).resolve().parent.parent
ANALYSIS_DIR = EXP_DIR.parent
SIBLING_DIRS = [
    ANALYSIS_DIR / "01_monot5_layer_patching",
    ANALYSIS_DIR / "03_monot5_head_patching_ablation",
    ANALYSIS_DIR / "06_monot5_query_doc_patching",
    ANALYSIS_DIR / "11_monot5_encoder_head_patching",
    ANALYSIS_DIR / "13_monot5_encoder_decoder_path_patching",
    ANALYSIS_DIR / "14_monot5_heldout_mean_diff_defense",
]
for _p in [*SIBLING_DIRS, EXP_DIR]:
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
