# Experiment 6 — Query/Document Positional Patching + Attention Analysis

Asks a question neither Experiment 1 (layer-level) nor Experiment 3
(per-head) can: does the keyword-stuffing attack, injected into the
**document** field, leak into the encoder's representation of the
**query**? If so, that would be document→query cross-contamination inside
the encoder — a different failure mode than "the document just looks more
relevant."

Builds on Experiments 1 (`../01_monot5_layer_patching/`) and 3
(`../03_monot5_head_patching_ablation/`): model loading, scoring
(`logit(true) − logit(false)`), the 105-attack grid, the padded-control
(Type B) construction, and the fwd/rev/combined effect formulas are
imported from `src/`, not copied. Experiment 3's folder is untouched and
not imported from — Experiment 6 is self-contained.

## What "query span" and "document span" mean

The template is `"Query: {query} Document: {passage} Relevant:"`. Spans are
found by building probe substrings ("Query: ", "Query: {query}",
"Query: {query} Document: ", "Query: {query} Document: {passage}") and
verifying each is an exact token-level prefix of the full prompt —
alignment-checked the same way Experiment 1 checks attack insertion
positions (`src/alignment.py`), failing (and counting) rather than
guessing on a SentencePiece boundary mismatch. See `exp6lib/spans.py`.

**Document span includes the injected attack tokens.** The attack is
textually inserted *inside* the document field (`attacked_passage` is
substituted whole into `{passage}`), so document-only patching, by
construction, patches the *entire* difference between control and attack —
this is why it's expected to trivially reproduce the attack (a sanity
check, not a finding). Query span is always upstream of the document field
and is identical across Type A (clean), B (control), and C (attack)
encodings; only the document span's end shifts by the attack's inserted
token count between clean and control/attack indexing.

## Part 1 — Attention analysis (descriptive)

Per encoder layer, on the **clean** and **attacked** inputs (not the padded
control — this is about how attention actually behaves on real inputs):

```
normalized_mass(A -> B) = mean_{i in A}[ sum_{j in B} attn[i,j] ] / (|B| / total_seq_len)
```

averaged over heads (per-head is out of scope this pass, matching Part 2).
`> 1` means region A attends to region B more than a uniform baseline
would predict.

## Part 2 — Positional patching (causal)

Three conditions — `doc_only`, `query_only`, `both` — patch the attack run's
activations into the control run (Type B) at document-span positions,
query-span positions, or their union, leaving all other positions as the
base run's own value.

- **Encoder self-attention** (whole block, all heads): a forward hook
  blends, at masked positions, the *other* run's cached contribution with,
  at unmasked positions, the *current* run's own contribution — computed
  inside the hook exactly as `src/activation_hooks.make_cache_hook` does.
  A mask covering the whole sequence reproduces Experiment 1's whole-layer
  patch bit-for-bit (tested).
- **Decoder cross-attention** (encoder-row swap): monoT5's decoder passes
  `key_value_states=encoder_hidden_states` as a **keyword** argument
  (verified against the installed transformers' `T5Block.forward` source),
  so patching requires `register_forward_pre_hook(..., with_kwargs=True)`
  on one decoder layer's cross-attention module — every other decoder
  layer keeps reading the base run's own, unpatched encoder output.

fwd/rev/combined effects are the same formulas as Experiment 1 (Eq. 4–6),
applied per (layer, condition) instead of per (layer, component).

## Timing pilot — read before running the full grid

`scripts/01_run_pilot.py` (1 attack, `relevant_start_5`, n=5, all layers,
all 3 conditions, both regions) **must** be run before
`scripts/02_run_grid.py`. Encoder self-attention patching requires the
encoder to fully re-execute per patch (the intervention happens mid-encoder,
same cost structure as Experiment 1); decoder cross-attention reuses one
encoder forward pass per input for every patch (same exact speedup
Experiment 3 uses). These two regions have very different costs, which is
exactly why the pilot measures them separately.

**Pilot result (this run, CPU, 5 examples):**

| region | s/example | ms/cell (36 cells) |
|---|---|---|
| encoder_self_attn | 69.7 | 1936 |
| decoder_cross_attn | 14.6 | 407 |
| attention analysis | 1.3 (fixed) | — |

Extrapolated full-grid wall-clock (CPU, same throughput as the pilot):

| grid size | total examples | estimated time |
|---|---|---|
| (a) Exp-3 tiering: 10×105 + 100 canonical | 1,150 | **27.4 hours** |
| (b) Higher-n: 30×105 + 100 canonical | 3,250 | **77.3 hours** |

These are CPU numbers (the pilot ran on the login node, which has no GPU,
since it was small enough to be feasible directly). Full-grid runs in this
codebase use a SLURM GPU job; based on Experiments 1 and 3's measured
CPU→GPU speedups on this cluster (roughly 8–15×), a rough GPU estimate is
**~1.8–3.4 hours for (a)** and **~5.2–9.7 hours for (b)** — extrapolated,
not measured; a short GPU-side pilot re-run would tighten this if precision
matters before committing a large allocation.

**Grid size confirmed:** Exp-3 tiering (10 examples × 105 attacks + 100 on
the canonical attack), launched on GPU via SLURM — see `configs/default.yaml`.

## Usage

```bash
conda activate advseq2seq
cd patch_code/analysis/06_monot5_query_doc_patching

bash bash/run_tests.sh          # unit tests (real tokenizer + tiny random T5, CPU)
bash bash/run_pilot.sh          # timing pilot — run this first

# after confirming configs/default.yaml's runs.grid.n_examples:
bash bash/run_all.sh            # full pipeline (resume-safe), or:
bash bash/run_all.sh configs/default.yaml --run grid   # one run at a time
```

SLURM: see the header of `bash/run_all.sh`.

## Outputs

```
outputs/                        # full grid (not yet run)
outputs_pilot/                  # pilot run (5 examples, relevant_start_5)
  pilot_rows.csv                # raw per-cell rows, for manual inspection only

outputs/{grid,canonical}/attacks/{attack}/
  results.csv                   # PRIMARY: one row per (layer, region, condition, example)
  selected_examples.jsonl       # exact examples used
  status.json                   # resume/status bookkeeping
outputs/{grid,canonical}/aggregated/
  layer_condition_attack_summary.csv   # derived: mean±std per (region, attack, layer, condition)
  layer_condition_summary.csv          # derived: mean±std per (region, layer, condition), pooled
  attention_summary.csv                # derived: mean±std attention mass per (attack, layer)
outputs/plots/
  attention_mass_clean_vs_attack.png      # 1. normalized attention mass, clean vs attack
  effect_by_condition.png                 # 2. score effect by condition, per layer, per region
  query_only_effect_across_attacks.png    # 3. query-only effect across 105 attacks (box plot)
```

## Deliberately not implemented yet

Per-head granularity for either region — deferred until whole-block
results show a split worth localizing further (same deferral pattern as
Experiment 3's joint-head ablation).
