# Experiment 13 ("2A") — Encoder-head -> decoder-cross-attention-head path patching

Sections 4.5 (Experiment 11) and 4.6 (Experiment 3) separately identify a
sparse set of encoder self-attention heads that build the attack-related
representation and decoder cross-attention heads that read it. Neither
establishes that a *specific* encoder head's contribution reaches the
score *through* a *specific* decoder head — head importance is not the
same claim as head-to-head connectivity. This experiment answers:

> Do specific encoder heads causally transmit the adversarial attack
> signal through specific decoder cross-attention heads?

It recovers the coarse pathway `encoder head A -> decoder head B ->
relevance score` for the 18 x 31 = 558 candidate paths formed by the
already-identified important heads, over all 105 attacks. It does **not**
restrict the sender to specific token positions (that is Experiment 2B,
which will combine this experiment's strongest/stablest paths with
Experiment 1's query-token localization results).

## Head sets and their provenance

**18 encoder senders** (`configs/heads/encoder_senders.json`): the
encoder self-attention head-slots whose mean combined effect exceeded
0.02 on Experiment 11's canonical run (single attack `relevant_start_5`,
n=100) — `11_monot5_encoder_head_patching/outputs/head_summary_canonical.csv`.
This reproduces the report's "18 of 144 head-slots" figure exactly.

**31 decoder receivers** (`configs/heads/decoder_receivers.json`): the
top-31 `decoder_cross_attn` head-slots by `combined_effect_mean` in
Experiment 3's Grid A (105 attacks, n=100) —
`03_monot5_head_patching_ablation/outputs/grid_a/aggregated/head_summary.csv`,
taken by **rank**, not by a 0.02 threshold.

**Why rank, not the report's stated 0.02 threshold.** The Experiment 3
report claims 33/288 head-slots (31 cross-attn + 2 self-attn) exceed 0.02
at n=100. That number does not reproduce from any on-disk aggregate:

| source | heads > 0.02 | notes |
|---|---|---|
| flat `grid_a/aggregated/head_summary.csv` (n=10432) | 22 (21 X + 1 S) | this is *exactly* Experiment 2's `flagged_heads.json` — the "22-head steering subset" this experiment was told **not** to reuse |
| attack-balanced Grid A (mean of 105 per-attack means) | 21 | — |
| Grid B (canonical single attack) | 11, all X | — |

`L10-S-H3`, which the report lists as a self-attention exception at
+0.021, is actually **-0.031** in the on-disk flat aggregate. Since the
report's 33/31 figure is not reproducible and the reproducible
alternative (22 heads) is explicitly off-limits, the receiver set here
is instead the top-31 `decoder_cross_attn` heads by rank in the flat
Grid-A aggregate — fully reproducible from data already in the repo, and
confirmed by the user (2026-08-09) as the resolution. Self-attention
heads are excluded entirely per the task spec. Re-derive both lists any
time with `scripts/00_derive_head_lists.py` (read-only w.r.t. Experiments
3/11's own outputs).

Both lists are asserted at runtime (`exp13lib/head_lists.py`):
18 senders x 31 receivers = 558 candidate paths.

## What "true path patching" means here, and how it's implemented

Patching sender A alone (Experiment 11) or senders and receivers
independently only shows both matter, not that A's effect reaches the
score *through* B. Path patching isolates the single route
`A -> encoder -> B -> downstream decoder -> score` while every other
encoder->decoder cross-attention route is held at baseline.

**Step 1 — build the hybrid encoder representation.** Starting from the
base run's own encoder computation (padded control for the forward
direction, attacked input for reverse), splice sender head A's `.o`-input
slice from the *other* run at A's own layer (all sequence positions —
Experiment 2A is coarse, not position-restricted), then let the rest of
the encoder process that splice normally. This reuses Experiment 11's own
per-head hook mechanism (`exp11lib/head_hooks.py`) unchanged; the only
addition (`exp13lib/path_engine.py::build_encoder_hybrid`) is returning
the *propagated final hidden state* instead of reading off a score.
Computed once per sender per example, reused across every receiver.

**Step 2/3 — feed the hybrid to receiver B only.** T5 multi-head
attention has no cross-head interaction until the final concatenation +
output projection (`.o`): each head's output depends only on its own
per-head Q/K/V, never on any other head's K/V source. That means
"receiver B alone reads the hybrid K/V, every other head at B's layer
reads baseline K/V" is *exactly* equal to "the whole layer reads hybrid
K/V, then keep only B's slice of the result" — the other heads' baseline
outputs are simply discarded, not altered. So, per (sender A, receiver
layer L, direction):

- **Step B** (`compute_receiver_layer_step_b_cache`): one decoder forward
  pass over the baseline encoder state (control for forward, attack for
  reverse) with layer L's cross-attention `key_value_states` swapped to
  the hybrid representation via a forward pre-hook on
  `model.decoder.block[L].layer[1]` (the `T5LayerCrossAttention` module,
  *not* the `.o` projection — `key_value_states` arrives there as a
  keyword argument from `T5Block.forward`). Every other layer, earlier
  and later, is untouched, so the decoder hidden state feeding layer L's
  query projection is identical to an ordinary unpatched pass. Caches the
  resulting `.o`-input at layer L (all 12 heads at once).
- **Step D**: Experiment 3's own per-head batched decoder pass
  (`headlib.engine.head_scores_for_slot`), using Step B's cached tensor
  as the "replacement" and the standard block-diagonal per-head mask.
  This reproduces, for every head index at layer L in one batched pass,
  "baseline everywhere except head h's slice <- Step-B's head-h slice",
  then lets the decoder continue normally past layer L (later layers,
  final layernorm, `lm_head`) — downstream propagation is real, not
  truncated. **Crucially, Step D never swaps `key_value_states` anywhere**
  — the hybrid signal only ever appears as a pre-computed value grafted
  onto one head's `.o`-input slice in the one scored pass. This is
  verified explicitly in `scripts/01_sanity_checks.py` (check 4) by
  instrumenting all 12 decoder layers' `key_value_states` during the
  scored pass and asserting every one of them is the plain baseline
  tensor.

Step B is cached once per (sender, receiver **layer**, direction) and
reused for every receiver head at that layer (several of the 31
receivers share a layer); Step D naturally batches over all 12 heads of
that layer in one pass. No sender-hybrid encoder pass or Step-B pass is
ever recomputed across receivers, per the task's caching requirement.

### Forward / reverse / combined effect

```
delta = score_attack - score_control          (skip example if |delta| < 1e-4)
path_forward(A,B)  = (score_path_forward(A,B)  - score_control) / delta
path_reverse(A,B)  = (score_attack - score_path_reverse(A,B))   / delta
path_combined(A,B) = min(path_forward(A,B), path_reverse(A,B))
```

Same normalization/definitions as Experiments 3 and 11. Values are never
clipped; negative values are preserved.

## Attack grid, filtering, and example selection

All 105 attacks (7 tokens x 3 positions x 5 repetitions), inherited from
`01_monot5_layer_patching/configs/multi_attack.yaml`. Padded-control /
attacked encodings use the position-agnostic general alignment
(`src.model_utils.build_padded_control_and_attack_encodings_general`),
so start/end/random attacks are all handled the same way.

Successful-instance filter: `delta = score_attack - score_control > 1e-4`
(`src.patching.SKIP_EPSILON`), evaluated on **this experiment's own live
score computation**, per example — not trusted from any stored value.
`run_example_paths` returns `None` for an example that fails this check,
and no rows are ever written for it.

Example *sourcing* reuses Experiment 1's per-attack selection pool
(`01_monot5_layer_patching/outputs/attacks/{attack}/scores/selected_examples.jsonl`),
pre-filtered on the file's stored `attack_delta_vs_control > 1e-4` to
build a cheap candidate pool (no attack-level mean-effect filter — both
positive- and negative-mean-effect attacks are eligible, per instance).
Each attack's qualifying pool is shuffled once with a per-attack-seeded
RNG (`seed:attack_name`, seed=42) and truncated to
`run.max_examples_per_attack` — so an n=30 run's examples are exactly an
n=10 run's 10 plus 20 more (not a fresh re-shuffle), while remaining a
genuine random sample rather than a top-delta ranking. Change only
`run.max_examples_per_attack` in `configs/default.yaml` to move between
n=10 (initial breadth run), n=30, and n=100 — no code changes.

## Compute / caching

Per example: 2 baseline encoder passes (control, attack) with per-layer
`.o`-caching restricted to the 5 distinct sender layers {0,8,9,10,11}
(`exp13lib.path_engine.cache_encoder_and_score`, adapted from
`exp11lib.engine.cache_head_inputs` to also return the final hidden
state); per sender (18), 2 encoder-only forward passes to build
`encoder_hybrid_forward`/`encoder_hybrid_reverse`; per (sender, receiver
layer in {0,4,5,6,7,8,9,10,11}, direction), one Step-B decoder pass +
one batched Step-D decoder pass. All decoder passes are single-token
(monoT5 scoring never generates), so this stays cheap even though the
path count is large.

## Sanity checks (`scripts/01_sanity_checks.py`)

Run `bash bash/run_debug.sh` before launching the full sweep. It runs
`scripts/00_debug_single_path.py` (1 attack, 1 example, 1 sender, 1
receiver — prints baseline/path scores and a cross-check against
Experiment 11's own per-head patch) followed by all required checks:

1. **No-change control** — hybrid built from the base run's own
   activation (a true no-op) reproduces the baseline score exactly.
2. **Sender-only reconstruction** — "whole decoder reads the hybrid"
   (no receiver isolation) matches Experiment 11's own per-head patch
   score for that sender, within float tolerance.
3. **Full encoder control** — an unpatched decoder pass over the plain
   control/attack encoder state reproduces `score_control`/`score_attack`.
4. **Single-receiver isolation** — instruments `key_value_states` on all
   12 decoder cross-attention layers during the scored Step-D pass;
   asserts every layer receives the plain baseline tensor (the hybrid
   never appears there — see architecture note above).
5. **Correct head sets** — 18 senders, 31 receivers, 558 candidate paths.
6. **Successful-instance filtering** — a synthetic zero-delta example
   (control vs. itself) is confirmed skipped.
7. **Sample cap** — selection never exceeds `max_examples_per_attack`.
8. **Forward/reverse baseline preservation** — same evidence as check 4:
   Step D's decoder base is always the control encoder state for forward,
   attack for reverse, before receiver B is touched.

Results are printed and written to `outputs/diagnostics/sanity_check_report.txt`.

## Running

```bash
conda activate advseq2seq
cd patch_code/analysis/13_monot5_encoder_decoder_path_patching

# 1. Debug + sanity checks (always run first)
bash bash/run_debug.sh

# 2. Full sweep (105 attacks x up to 10 examples, checkpointed/resumable)
bash bash/run_sweep.sh                       # runs sweep + aggregate + plots
bash bash/run_sweep.sh --max-examples 30     # later n=30 pass, same config
```

`scripts/02_run_sweep.py` writes one raw CSV + `status.json` per attack
under `outputs/raw/{attack_name}/`; an already-completed or
already-skipped `(qid, docid)` is never recomputed on re-run, so the
sweep can be killed and restarted safely (SLURM preemption etc. — see
`bash/run_sweep.sh`'s `run_job.sh` example for the cluster invocation).

## Outputs

```
outputs/
  raw/{attack_name}/paths.csv, status.json   -- per-example, per-(A,B) rows
  aggregates/
    per_attack_sender_receiver.csv            -- Table 1
    global_sender_receiver_long.csv           -- Table 2 (long form)
    global_matrix_{combined,forward,reverse,stability}.csv  -- Table 2 (18x31 wide)
    sender_summary.csv                        -- Table 3
    receiver_summary.csv                      -- Table 4
    path_stability.csv                        -- Table 5
  plots/
    path_heatmap_combined.png                 -- Plot 1 (headline figure)
    path_heatmap_forward.png                  -- Plot 2
    path_heatmap_reverse.png                  -- Plot 3
    path_stability.png                        -- Plot 4
    sparse_circuit_topk.png                   -- Plot 5 (optional, top-25 |combined| edges, visualization-only)
  diagnostics/
    sanity_check_report.txt
    path_heatmap_combined_clustered.png       -- secondary, value-clustered ordering
```

No importance threshold (e.g. `path_combined > 0.02`) is applied
anywhere in this experiment — all continuous aggregates are saved as-is
for interpretation after inspecting the distribution (mean, median,
forward/reverse agreement, cross-attack consistency), per the task spec.

## Relationship to Experiments 1 and 2B

Experiment 1 (running independently) identifies which query-word
positions causally carry the attack effect. Experiment 13 identifies
which encoder heads feed which decoder heads, coarse (all encoder
positions, not position-restricted). Experiment 2B will intersect the two:
for the strongest/stablest `A -> B` paths found here, repeat path
patching restricting sender A's intervention to the token positions
Experiment 1 identifies as important, i.e.
`encoder head A @ important query-word position/group -> decoder head B
-> score`. This experiment's sender/receiver IDs and raw per-example path
results (`outputs/raw/`) are the reusable input to that follow-up.
