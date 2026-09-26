# Experiment 11 — Design Decisions

Deliberate choices not fully obvious from the code, in the order a reader would hit them.

## 1. `exp11lib` is a fresh package, not an import of Experiment 3's `headlib`

Project convention (see Experiments 2/7/9/10): each experiment folder is self-contained;
another experiment's lib package is only imported when a prompt explicitly calls for reuse
of *that specific* logic (here: Experiment 6's span-finding and attack-token exposure).
Experiment 3's per-head mechanism is **structurally** reused (same `.o`-input slicing,
same block-diagonal mask, same mean==rev identity) but reimplemented for the encoder
rather than imported, since the decoder and encoder cases differ in exactly the ways that
matter for correctness (sequence length, encoder-output-reuse eligibility — see below).

## 2. No "component" axis

Experiment 3 scores `{decoder_self_attn, decoder_cross_attn}` × layer × head. The encoder
has exactly one attention sublayer per block (`T5Block.layer[0].SelfAttention`;
`layer[1]` is the feed-forward, no heads), so scope is `(layer, head)` only — 144 slots,
not 288.

## 3. Whole-sequence patching, not per-position

Step 1/2 patch a head's `.o`-input slice at **every** sequence position at once (matching
Experiment 6's whole-block "both"/`doc_only`/`query_only` conditions' granularity, not its
per-position template-token patching). A joint head×position sweep would be
12 layers × 12 heads × ~150 positions — explicitly out of scope per the experiment prompt
("this is expected and is a feature, not a complication — see Step 4", which instead
cross-references *attention*, not a full causal head×position sweep).

## 4. No encoder-output-reuse trick (unlike Experiment 3)

Experiment 3's headline speed trick — run the encoder once, reuse its output for every
decoder-side patch — requires the patch to happen strictly downstream of the encoder.
Encoder self-attention patches happen **inside** the encoder's own forward computation
(same limitation `exp6lib/engine.py`'s whole-block encoder patching has), so every
(layer, direction) unit re-executes the encoder. The batched-over-heads trick still
applies (see `exp11lib/engine.py` docstring): one batched (batch=12) encoder+decoder pass
scores all 12 heads of a layer at once, instead of one pass per head.

## 5. Layer processing order vs. layer scope

`configs/default.yaml`'s `heads.layer_priority` orders layers 9→11→8→...→0 for
**processing order** only (front-loads the range three independent prior results point
to), not a sub-selection — `heads.layers: null` still means all 12 layers run and are
written. In practice this experiment's per-example cost turned out low enough
(~1.6–2.6s/example at 144 head-slots × 2 ablation methods, measured directly) that
running 0–8 in full was never actually a compute concern; the ordering is kept anyway
since it costs nothing and front-loads the informative range in the output logs.

## 6. Mean/control ablation is primary; zero is secondary

Per Experiment 3's own finding (zero ablation over-attributes importance vs. mean/control
ablation), every Step 1/2 table, flag threshold, and headline number in this experiment
uses `combined_effect` (built from fwd/rev patching, which is mean/control-ablation-
equivalent by the same identity Experiment 3 relies on: `score_ablated_mean ==
score_patched_rev` on the attack base, computed once). Zero ablation is retained only for
the Step 1 zero-vs-mean scatter, matching Experiment 3's Figure 6.

## 7. Flagging: two different definitions, used for two different purposes

- Steps 1/4/5 flag a `(layer, head)` **slot** using the **canonical** attack's (n=100)
  per-slot mean `combined_effect > flag_threshold` (0.02), restricted to layers 9–11 for
  Step 5 (the late-layer range every prior result points to) — this is the most
  statistically solid single-run estimate and gives a slot-level (not just head-index-
  level) flag, which Step 4/5 need.
- Step 2/3 flags a `head_idx` (pooled across layers 9–11) using the **full 105-attack
  sweep**'s peak — this is the robustness check ("X of 105 attacks") and feeds the
  additivity sum.

Both are reported; neither silently overrides the other.

## 8. Attack-token position: reused, not recomputed

Step 5 needs the exact inserted-attack-token indices. Rather than re-deriving them from
`src/alignment.py`, this experiment reuses `exp6lib.run_utils.ExampleInputs
.attack_span_indices` — a field added to Experiment 6's `run_utils.py` during the
template-token extension specifically to expose `AlignmentResult.inserted_positions`
(previously computed internally but never surfaced; Experiment 10's README notes this gap
explicitly). No new alignment logic was written for this experiment.

## 9. Step 5's random-token control

"A random non-attack document token at a matched position" is implemented as: sample
`len(attack_span_indices)` token indices from the document span, excluding the attack
span, seeded deterministically per `(global seed, qid:docid)` (`exp11lib/hub_analysis
.pick_random_control_positions`) — matched in **count**, not exact relative offset (attack
spans vary in length across attacks/repetition counts, so an exact-offset match isn't
well-defined project-wide). Reproducible given the same seed and example set.

## 10. Step 4: per-head attention was not available to re-slice

Checked directly: Experiment 6's `outputs/canonical/aggregated/attention_summary.csv` has
no head column — `normalized_attention_masses` averages over heads before any row is
written, so no per-head tensor survives to disk. Step 4 recomputes attention fresh, on the
*same* canonical-attack example set Step 1 used (loaded from
`outputs/canonical/selected_examples.jsonl`), on both clean and attacked inputs (matching
Experiment 6 Part 1's own clean-vs-attack convention, not the padded control — Part 1 is
descriptive, not a causal-patching baseline).

## 11. Classification thresholds

"High mass" = normalized mass > 1.0 (the uniform-attention baseline, same convention as
`exp6lib/attention.py` throughout this project). "High causal effect" = combined_effect >
0.02 (`flag_threshold`, matching Experiment 6's template-position classification for
project-wide consistency). Both are configurable in `configs/default.yaml`, not hardcoded
in the classification logic.
