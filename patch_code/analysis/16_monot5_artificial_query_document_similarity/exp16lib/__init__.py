"""
exp16lib — Experiment 16: artificial query-document similarity.

Observational representation analysis (NO patching, NO ablation, NO
steering): at 25 encoder checkpoints (embedding + post-attention and
post-MLP residual states of every encoder layer) we mean-pool the query-text
tokens and the document tokens separately and take the cosine of the two
pooled vectors. This is compared across

  * clean Type-A pairs            vs the cached monoT5 score      (RQ1)
  * TREC DL19 qrel 2/3 vs qrel 0  within query                    (RQ2)
  * attacked vs padded control    all 105 attacks, no success filter (RQ3)
  * attack-level delta_sim        vs attack-level delta_score     (RQ4)

Earlier experiments are imported, never copied (project reuse convention):
  Exp 01  src.*        model loading, prompt format, padded-control
                       construction + token alignment, attack registry,
                       Exp 01 scoring functions (smoke score-cache check)
  Exp 03  headlib.*    config loading with attacks.inherit_from
  Exp 06  exp6lib.*    query/document span detection (template exclusion)
  Exp 13  exp13lib.*   canonical important-head lists (18 encoder, 31 decoder cross-attn)
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
    ANALYSIS_DIR / "13_monot5_encoder_decoder_path_patching",
    ANALYSIS_DIR / "14_monot5_heldout_mean_diff_defense",
]
for _p in [*SIBLING_DIRS, EXP_DIR]:
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
