# Experiment 10 — Encoder-Output Difference Geometry: Design Decisions

## Mean-centering: NOT applied, per-attack, before SVD

The task's core question is whether the attack-control encoder diff is
"a single low-rank direction (consistent across attack tokens, i.e.
defensible with one global steering vector)." A steering-vector defense
adds one fixed vector to (or subtracts it from) the encoder output — the
relevant test is whether the RAW diff vectors themselves are dominated by
one direction, not whether the *variation around their mean* is low-rank.

Mean-centering before SVD would answer a different, subtler question (is
the residual after removing the per-attack mean diff low-rank?) and could
actively hide the signal this experiment is looking for: if every diff
vector for an attack points almost the same way, centering removes exactly
that shared direction before SVD ever sees it, making a genuinely
low-rank/single-direction effect look artificially high-rank in the
centered analysis.

**Decision**: `exp10lib/geometry.compute_pca_summary` runs `np.linalg.svd`
on the raw, uncentered per-attack diff matrix. The per-attack mean diff
vector is computed and reported separately (`mean_direction` — itself the
Experiment-2-style candidate steering vector for that attack) and used
only to fix the sign ambiguity of the top singular vector (see below), not
to center the data before decomposition.

**Per-attack, not global.** Each PCA/SVD is run on one attack's own pooled
diff matrix (all examples, all positions, for that attack only), never
pooled across attacks — pooling across attacks before SVD would conflate
"is this attack's own diff low-rank" with "do different attacks share a
direction," which is exactly the separate question Step 4 (cross-attack
cosine similarity) is designed to answer using each attack's *own*
already-computed top-1 direction. Keeping the two steps separate (per-attack
SVD, then cross-attack comparison of the results) avoids answering both
questions with one number.

## Differing sequence lengths: ragged accumulation, no padding/truncation

Examples for one attack have different query/passage lengths, hence
different total token counts. The per-attack diff matrix's rows are
individual `(position, example)` pairs — not `position index` treated as a
shared feature dimension across examples — so there is no cross-example
alignment requirement to satisfy, and therefore no padding/truncation
decision that could bias the matrix toward short examples.

**Decision**: `scripts/01_extract_diffs.py` accumulates each example's own
`(seq_len_i, d_model)` diff array into a Python list per layer, then
`np.concatenate`s all examples' arrays along axis 0 at the end. A
100-token example contributes 100 rows; a 250-token example contributes
250 rows. No example is truncated, no example is padded, and no example's
positions are silently dropped. (The only per-example alignment that
matters — control and attack sequences having identical length and
position-for-position correspondence — is guaranteed by construction in
Experiment 1's `build_padded_control_and_attack_encodings_general`, not by
anything in this experiment.)

One consequence worth flagging: longer examples contribute proportionally
more rows to the PCA input than shorter ones. This is judged correct, not
a bias to correct for — a position is a position; weighting every example
equally regardless of length would be the artificial choice, effectively
down-weighting long documents' many `other_document` positions relative to
short documents' few.

## Sign convention for PCA/SVD sign ambiguity

`np.linalg.svd`'s singular vectors have an arbitrary overall sign (`v` and
`-v` are equally valid top components). Comparing "top-1 directions" across
attacks via cosine similarity is meaningless without first fixing this
sign consistently.

**Decision**: for each (attack, layer), the top singular vector is flipped
if necessary so that `dot(top_direction, mean_direction) >= 0`, where
`mean_direction` is that same attack's own mean diff vector over the
identical population of rows used for the SVD (Step 4's task
specification: "align signs ... by requiring positive dot product with the
mean diff vector"). This is well-defined and attack-local — it does not
depend on any other attack's data — so the resulting cross-attack cosine
similarities are directly comparable.

## `injected_attack_token` tagging accounts for SentencePiece re-tokenization

Naive whitespace/word-boundary tagging would be wrong here: SentencePiece
can re-tokenize the passage differently depending on what immediately
precedes it (a documented failure mode Experiment 1's alignment code
already guards against — see `src/alignment.py`'s docstring on why the
prefix-walk + suffix-check + `difflib` fallback exists at all).

**Decision**: injected-token positions are taken directly from Experiment
1's `align_attack` (`src/alignment.py`), specifically its
`inserted_spans` field — the same alignment-checked utility already used
to build the padded-control encoding itself, so "where the pad tokens are
in Type B" and "where the injected tokens are tagged in Type C" are
guaranteed consistent by construction (both come from the same
`AlignmentResult`).

**Gap this fills, relative to Experiment 6**: Experiment 6's
`find_query_and_doc_spans` (`exp6lib/spans.py`) already calls
`build_padded_control_and_attack_encodings_general` internally (hence
`align_attack`) but only ever consumes the *scalar count*
(`align_result.n_inserted`) to widen its document span
(`doc_span_control_attack = (doc_start, doc_end + n_attack_tokens)`) — it
never surfaces `inserted_spans` (the exact sub-range) to callers. Since
Experiment 10 needs the precise sub-span (not just the count), it calls
`align_attack` itself (via Experiment 1's existing
`build_padded_control_and_attack_encodings_general`, which returns the
full `AlignmentResult`) rather than adding a new return value to
Experiment 6's function or re-deriving alignment from scratch. See
`exp10lib/tagging.py`'s module docstring for the same explanation next to
the code.

## Why raw `output_hidden_states=True`, not Experiment 1's activation-hook cache

Experiment 1's `src/activation_hooks.py` caches each sub-layer's residual
**contribution** (`out_hidden - in_hidden`), built for activation patching.
This experiment needs the fully-accumulated hidden state at a given
encoder depth (the same quantity DecoderLens, Experiment 8, reads out) —
a different physical tensor. Reusing Experiment 1's cached `.pt` files
would also only cover the `relevant_start_5` example pool (~100 examples;
every other attack's cached activations are deleted after Stage 03 by the
multi-attack pipeline's cleanup step), not the 15 attacks this experiment
needs. `model.encoder(..., output_hidden_states=True).hidden_states[L]` is
therefore the correct and only practical source, computed fresh per
example (cheap: encoder-only forward passes, no candidate re-ranking).

## Attack selection: reuse Experiment 1's `mean_attack_delta` ranking directly

Rather than re-scoring all 105 attacks to rank them, `exp10lib.run_utils
.select_attacks` reads Experiment 1's already-computed
`outputs/attack_comparison/summary.csv` (`scripts/11_compare_attacks.py`'s
output) and sorts by `mean_attack_delta` descending — the same ranking
metric named in the task spec. Rows are cross-checked against the
`attacks.inherit_from`-resolved curated 105-attack grid (same mechanism
Experiments 2/3/6/7 use) so a stale or hand-edited summary CSV can't
silently select an attack outside that grid; this makes the
`attacks.inherit_from` block functionally load-bearing (a validation set),
not just present for documentation consistency with the other experiments.

## Sample size: primary (15 attacks × 100 examples) run first, optional full-grid extension available

`configs/default.yaml` implements the required primary scope. `configs/
full_grid.yaml` implements the task's explicitly optional extension (all
105 curated attacks) using the identical pipeline — only `selection.
n_attacks: null` differs from the primary config. `n_examples` was raised
from the task's suggested reduced depth (n=10, matching Experiment 6's
grid-a depth) to the full n=100 (matching the primary run's depth), since
the per-example cost here is a handful of encoder-only forward passes (no
100-candidate re-ranking, unlike Experiments 1/8/9) — cheap enough that
running the full 105-attack grid at full depth was preferred over a
reduced-depth compromise. Both configs were run for this report (see
Reproducibility section of the final `.tex` report).
