# Experiment 7 — Logit Lens on monoT5 Decoder Cross-Attention

Merges what would have been three separate analyses — a general logit
lens, the query/document/attack composition of top-k, and attack-token
rank/logit tracking — since all three need the identical underlying
machinery: capture a cross-attention contribution, project it through the
model's own unembedding, read off top-k. See `DECISIONS.md` for the full
rationale behind every design choice below; this README is the operational
summary.

Builds on Experiments 1 (`../01_monot5_layer_patching/`) and 3
(`../03_monot5_head_patching_ablation/`): model loading, scoring, the
105-attack grid, the padded-control construction, and Experiment 3's
per-head importance results (read live, not hardcoded) are reused, not
duplicated. No other experiment folder is modified or imported from.

## Scope: cross-attention only

Not decoder self-attention, not the full residual stream — a direct
extension of Experiments 1 and 3's finding that the attack signal
concentrates in decoder cross-attention. Revisit only if these results are
ambiguous (see `DECISIONS.md`).

## What gets projected

At each decoder layer, the cross-attention sublayer's **contribution**
(`out_hidden - in_hidden` — what it writes into the residual stream, not
the accumulated hidden state) is projected through `model.lm_head` with
T5's own `scale_decoder_outputs` convention (`d_model**-0.5`), deliberately
**without** re-applying `final_layer_norm` — this is direct logit
attribution of one component, not classical full-residual logit lens (see
`DECISIONS.md` for why re-applying layer norm to an isolated contribution
would be misleading).

- **Whole-block**: all 12 decoder layers, always.
- **Per-head**: only heads Experiment 3 flagged (`combined_effect_mean >
  0.02` for `decoder_cross_attn`, read live from Experiment 3's aggregated
  output). A head's contribution is `W_o(zero_pad(head_slice))` — exact,
  not approximate, because `W_o` has no bias (verified; per-head
  contributions sum exactly to the whole-block contribution, tested).

## The three analyses (one unified row per layer/head × run_type × example)

1. **General logit lens** — top-5/10/20 tokens and logits, for clean,
   control, and attack inputs.
2. **Query/doc/attack composition** — each top-k token (k=10 by default)
   tagged `attack` / `query` / `document` / `stopword` / `other`. Attack
   checked *first*: the attack-grid words ("relevant", "important", etc.)
   are ordinary English words that can coincidentally appear in query or
   document text, and undercounting that would understate the attack's
   apparent dominance.
3. **Attack-token tracking** — the specific injected token's rank and
   logit at every layer, logged for **clean, control, and attack** inputs.
   Clean/control give the near-zero baseline; without it, "the attack
   token reaches rank 3" has no reference point.

## Timing pilot — read before running the full grid

`scripts/01_run_pilot.py` (1 attack, `relevant_start_5`, n=5, all 12
whole-block layers + 12 Experiment-3-flagged heads, clean/control/attack)
**must** be run before `scripts/02_run_grid.py`. Whole-block is cheap (one
cached forward pass reused across all 12 layers); per-head is expensive
(each flagged head needs its own forward pass with a pre-hook on that
layer's `.o` projection) — this is exactly why the pilot measures them
separately.

**Pilot result (this run, CPU, 5 examples, 12 flagged heads found above the
0.02 threshold in Experiment 3's aggregated output):**

| component | s/example | ms/cell (36 cells) |
|---|---|---|
| whole-block | 3.78 | 105 |
| per-head | 34.24 | 951 |
| **total** | **38.02** | — |

Per-head dominates (~90% of total cost): each flagged head needs its own
forward pass with a pre-hook on that layer's `.o` projection, versus
whole-block's one cached forward pass reused across all 12 layers.

Extrapolated full-grid wall-clock (CPU, same throughput as the pilot):

| grid size | total examples | estimated time |
|---|---|---|
| (a) Same tiering as Exp 3/6: 10×105 + 100 canonical | 1,150 | **12.1 hours** |
| (b) Alternative, 30/attack: 30×105 + 100 canonical | 3,250 | **34.3 hours** |

These are CPU numbers. Based on Experiments 3 and 6's measured CPU→GPU
speedups on this cluster (10–35×, with Experiment 6 specifically seeing
~34× for its full-forward-pass-per-cell cost — structurally the closest
match to this experiment's per-head cost), a rough GPU estimate is
**~20–49 minutes for (a)** and **~1.0–2.3 hours for (b)** — extrapolated,
not measured.

**Grid size confirmed:** higher-n (30 examples × 105 attacks + 100 on the
canonical attack), launched on GPU via SLURM — see `configs/default.yaml`.

## Usage

```bash
conda activate advseq2seq
cd patch_code/analysis/07_monot5_logit_lens

bash bash/run_tests.sh          # unit tests (real tokenizer + tiny random T5, CPU)
bash bash/run_pilot.sh          # timing pilot — run this first

# after confirming configs/default.yaml's runs.grid.n_examples:
bash bash/run_all.sh            # full pipeline (resume-safe)
```

SLURM: see the header of `bash/run_all.sh`.

## Outputs

```
outputs/{grid,canonical}/attacks/{attack}/
  results.csv                   # PRIMARY: one row per (run_type, layer[, head], example)
  selected_examples.jsonl
  status.json
outputs/{grid,canonical}/aggregated/
  composition_summary.csv           # mean bucket % per (run_type, layer, attack), whole-block
  attack_token_trajectory.csv       # mean/std rank & logit per (run_type, layer, attack), whole-block
  per_head_token_frequency.csv      # top-token frequency per (run_type, layer, head-or-none)
outputs/plots/
  bucket_composition_by_layer.png       # 1. stacked bar, clean vs attack
  attack_token_trajectory.png           # 2. rank/logit promotion curve, clean/control/attack
  per_head_vs_whole_block.png           # 3. flagged heads vs whole-block, same layer
```
