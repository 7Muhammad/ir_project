# Experiment 14 — Decisions

Every deliberate deviation from, or choice not dictated by, the original
task spec. Read this before changing candidate-head sources, split logic,
position definitions, or scale selection.

## 1. Why 18 encoder + 31 decoder heads, not "18 + 33"

The task spec (and `current_report.tex`'s abstract, and Experiment 3's own
report) state the decoder side has "33 of 288 head-slots (31 cross-attention
+ 2 self-attention: L11-S-H3, L10-S-H3)" exceeding a 0.02 combined-effect
threshold. **This does not reproduce from any on-disk aggregate.**
Independently verified here (matching Experiment 13's own provenance note,
`13_monot5_encoder_decoder_path_patching/exp13lib/head_lists.py`):

- flat `grid_a/aggregated/head_summary.csv` (n≈10432, the current n=100-per-
  attack aggregate): **22 heads** > 0.02 (21 `decoder_cross_attn` + 1
  `decoder_self_attn`), and `L10-S-H3` is actually −0.031 there, not the
  +0.021ish the report implies.
- attack-balanced grid_a (mean of 105 per-attack means): 21 heads > 0.02.
- grid_b (canonical single attack only): 11 heads > 0.02, all cross-attn.

The 22-head flat-threshold set is *exactly* Experiment 2's
`flagged_heads.json` ("22-head steering subset"), which this experiment (per
the task's own instruction) was explicitly told not to reuse.

**Resolution:** since no on-disk aggregate reproduces "33/31", Experiment 14
reuses the same resolution Experiment 13 already settled on (per user
decision 2026-08-09, recorded in `exp13lib/head_lists.py`): 18 encoder heads
by threshold (reproduces the report's "18/144" claim exactly) + 31 decoder
`decoder_cross_attn` heads by **rank** (top-31 by `combined_effect_mean`),
excluding `decoder_self_attn` entirely. Total: 49 candidate heads, not 51.
See `scripts/00_derive_candidate_heads.py` / `exp14lib/head_lists.py`, which
assert these exact counts and fail loudly if they ever change.

## 2. Train/validation/test split rule

Split by unique `(qid, docid)` pair, **before** any attack-success
filtering (task spec section 3), using Experiment 1's own canonical
500-pair sample (`01_monot5_layer_patching/outputs/pairs/pairs.jsonl`)
rather than resampling. This file is a valid, reusable, attack-independent
pool because **every one of the 105 attack TSVs in the upstream ecir24 repo
shares the identical 42005-pair BM25 candidate universe** — verified
directly: `diff` of the `(qid, docid)` sets of two arbitrary attack files
(`associated_end_1` vs `associated_end_2`) gives set-equality, 42005/42005.
Split: seed 42, ratios 60/20/20, via a SHA-256-hash-based deterministic
shuffle (`exp14lib/splits.py::_stable_shuffle`) — not Python's built-in
`hash()`, which is randomized per-process and would make the split
non-reproducible across runs/machines.

## 3. Successful-instance threshold

`score_attack - score_control > 1e-4` (task spec section 4), matching
`src.patching.SKIP_EPSILON` used throughout the rest of the project.
Experiment 1's own `selected_examples.jsonl` files use a looser `> 0.0`
threshold; rather than rescoring, `exp14lib/data_pool.py` re-filters the
already-cached scores client-side to the stricter `1e-4` bound — free, no
extra forward passes.

## 4. Why encoder direction fitting still pools over all valid positions

Task spec section 5 is explicit: "position-specific direction fitting is
not part of this experiment." `exp14lib/direction_fit.py` reuses Experiment
2's `pool_activation` unchanged — encoder_self_attn activations are
mean-pooled over every `attention_mask==1` position before being folded
into the running mean, regardless of which of the 5 position masks will
later be used to *apply* the resulting direction. One head → one fixed
64-dim direction, full stop.

Directions are pooled across ALL 105 attacks' successful TRAIN instances
(round-robin-interleaved, see item 9), mirroring Experiment 2's grid_a
philosophy of a single direction per head rather than one direction per
(head, attack) — the task spec's "one fixed direction per head" (singular)
supports this reading.

## 5. Why position masks affect intervention location, not direction fitting

Direct consequence of item 4: the *direction* is a single 64-dim vector
computed once (item 4). The 5 *position masks* (document/query/template/
query_document/all_valid) only gate WHERE in the sequence that same frozen
vector gets subtracted (`exp14lib/hooks.py::make_encoder_masked_shift_pre_hook`).
This isolates "does WHERE we steer matter", holding the direction itself
fixed — exactly the question task spec section 6 asks.

## 6. Definition/source of template/query/document positions

Reused verbatim from Experiment 6, never redefined:
- query span, document span (Type B/C indexing, i.e. including injected
  attack-token positions): `exp6lib.spans.find_query_and_doc_spans`.
- The 7 named template/prompt sub-regions (`Query`, `:_after_query`,
  `Document`, `:_after_document`, `Relevant`, `:_after_relevant`, `</s>`):
  `exp6lib.template_positions.get_template_positions`, itself defined as
  the complement of query/document spans within the sequence.

Experiment 6 never collapsed those 7 sub-regions into one "template" mask
(it always consumed them individually or via the complement region
directly); `exp14lib/position_masks.py::build_position_masks` does that
collapse (`template_mask = union of the 7 groups' index lists`) — the only
genuinely new piece of position logic in this experiment, and it introduces
no new probe strings or boundary logic, only a union over Experiment 6's
own output.

`document` uses **Type B/C indexing** (`doc_span_control_attack`, which
includes the attack-token positions) for the attacked/control encoding, but
**Type A indexing** (`doc_span_original`, no attack tokens, shorter
sequence) for the clean-damage evaluation encoding — see
`exp14lib/position_masks.py`'s two entry points
(`masks_for_attacked_or_control` vs `masks_for_clean`).

## 7. Scale-selection rule

Highest MEAN normalized recovery on VALIDATION data; ties broken by the
smaller scale (task spec section 8, implemented literally —
`exp14lib/scale_selection.py::select_scales`). The validation
recovery-vs-clean-damage curve is computed and saved
(`outputs/scale_selection/validation_recovery_vs_damage_curve.csv`) for
inspection but never used to gate or reorder the selection itself, per the
task's explicit instruction not to bake in an arbitrary clean-damage
threshold.

The fixed reference scale (default 1.5, configurable) is always included in
every scale sweep (validation, IID test, attack-OOD), added to the
configured scale list if not already present, so its results are preserved
even if a smoke/custom config's scale list omits it.

## 8. Use of Type-B padded control as the attack-recovery target

Recovery (`exp14lib/metrics.py::recovery`) always measures distance to the
**padded control** (`score_control`, Type B — attack-token positions
replaced with masked pad slots), not the original clean input, matching
every prior causal experiment's (1/3/6/11/13) definition of the baseline
patching tries to explain, and matching task spec section 9's own formula
(`S_c` = control score throughout).

## 9. Use of Type-A original input for clean-damage evaluation

Clean-damage evaluation (task spec section 10) uses `original_score`
(Type A — the true unmodified clean prompt, no pad slots, no attack
tokens), reusing the value already cached in Experiment 1's
`selected_examples.jsonl` rather than re-tokenizing/re-scoring — this is
deliberately a *different* input type than the padded control (item 8),
since the question here is "does the defense damage the model's judgment
on a genuinely clean document", not "does it move the score back toward an
artificial pad-slot baseline".

## 10. Attack-OOD fold definitions

Three independent leave-one-factor-value-out partitions of the 105-attack
grid (`exp14lib/ood_folds.py`), each keyed off `AttackSpec.token` /
`.position` / `.repetitions` (already parsed by Experiment 1's
`attack_registry.py` from the attack filename): 7 token folds, 3 position
folds, 5 repetition folds. Held-out attacks are additionally pair-held-out
via the existing TEST split — the fold only controls which attack
CONFIGURATIONS are visible during fitting/selection, independent of which
PAIRS are visible. Per fold: direction fit on seen-attack TRAIN examples
only, scale selected on seen-attack VALIDATION examples only, evaluated on
held-out-attack TEST examples only — verified by
`tests/test_ood_folds.py::test_no_held_out_attack_examples_enter_the_pooled_training_set`.

## 11. Why multi-head defense is deferred to Phase 2

Directly stated in the task spec (sections 1, 2, 21): Phase 1 tests one
head at a time to isolate each individual head's/condition's causal
contribution before any joint/combined intervention is attempted; combining
heads introduces interaction effects (e.g. two heads' shifts could partially
cancel or compound) that are a separate, harder question requiring its own
methodology (relative scale calibration across heads, etc.) — not decided
here.

## 12. Recovery metric's ">1" interpretation is unreachable under the literal formula

Task spec section 9 lists `> 1` = "intervention moved beyond the control
score" as one of four named interpretations of `Recovery = (|S_a-S_c| -
|S_d-S_c|) / |S_a-S_c|`. Under this literal formula, since `|S_d-S_c| >= 0`
always, the numerator can never exceed the denominator, so `Recovery <= 1`
is a mathematical certainty — `Recovery > 1` cannot occur. Implemented
**exactly as specified** (no clipping, no reinterpretation) since the task
explicitly forbids clipping; this note just records that the `>1` branch of
the stated interpretation is dead code under the given formula, not a bug
in this implementation. See `exp14lib/metrics.py`'s docstring and
`tests/test_metrics.py::test_recovery_is_never_clipped_even_when_defense_overshoots`.

## 13. Train-example pooling policy (round-robin, not flat top-N)

Not prescribed by the task spec. When capping the TRAIN example pool used
for direction fitting (`sampling.max_train_examples_total`) and per-fold
OOD training pools, `exp14lib/data_pool.py::pool_round_robin` interleaves
one example at a time across attacks (sorted by attack name, highest-delta-
first within each attack) rather than taking a flat top-N sort by delta
across all attacks pooled together. A flat sort would let the single
attack with the most successful, highest-delta instances dominate the
capped pool; round-robin ensures the fitted direction reflects signal from
many attack configurations, not one. Fully deterministic given the same
input dict.

## 14. Efficiency scope: batch over scales, not over heads/masks/examples

Task spec section 12 asks to "batch scales/conditions where the current
architecture permits it without changing semantics." Implemented: every
decoder-head and every (encoder-head, position-mask) intervention batches
its full scale list into ONE forward pass (`exp14lib/hooks.py`, same
kernel-batching spirit as Experiment 3's batched-over-heads decoder
passes — real compute is NOT saved per se for the encoder side, since each
scale genuinely requires re-executing the encoder from the intervention
layer onward, but kernel-launch overhead is amortized across the batch).
Further batching across heads, position-masks, or examples was
deliberately NOT attempted — the encoder intervention happens INSIDE the
encoder's own forward computation (same limitation Experiments 6/11
already documented for encoder-side patching), so different heads/masks
require materially different hook state per forward pass; combining them
into a single batched call would require substantially more complex masking
logic for uncertain benefit at Phase-1 scope. This is a reasonable,
disclosed engineering boundary, not a spec violation.

## 15. Output directory nesting

Task spec section 14 lists required artifacts by name at what reads as the
top level of `outputs/` (e.g. `validation_results.*`, `iid_summary.*`).
This implementation nests them under per-stage subdirectories
(`outputs/validation/validation_results.csv`, `outputs/test/iid_summary.csv`,
etc.) so each stage's resume-safety `status.json` lives alongside its own
outputs, matching the project's established per-item `status.json`
convention (Experiment 1's `scripts/10_run_multi_attack_pipeline.py`). The
task spec itself calls the section-14 list "equivalent to" the required
outputs, not a literal path contract; `scripts/08_aggregate.py` additionally
produces flat, top-level `full_recovery_summary.csv` /
`full_clean_damage_summary.csv` consolidating everything into the
single-table form most likely useful for downstream reporting.
