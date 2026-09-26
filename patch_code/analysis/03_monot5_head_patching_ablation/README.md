# Experiment 3 — Per-Head Activation Patching + Ablation (monoT5 Decoder)

Localises the adversarial keyword-stuffing signal at **attention-head**
granularity: 12 decoder layers × {self-attn, cross-attn} × 12 heads =
**288 head-slots**. Decoder MLP is excluded (no heads); the encoder is out of
scope here (Experiment 1 covers layer-level encoder effects).

Builds on Experiment 1 (`../01_monot5_layer_patching/`): model loading, scoring
(`logit(true) − logit(false)`), the 105-attack grid (7 tokens × 3 positions ×
5 reps), the padded-control (Type B) construction, token alignment, and the
fwd/rev/combined effect formulas are all **imported** from `src/`, not copied.

## What "a head" means here

T5 attention computes `out = W_o · concat(head_0 … head_11)`. We intervene on
the input of the `.o` projection, where head *h* is the contiguous slice
`[h·d_kv : (h+1)·d_kv]`. `W_o` has no bias, so replacing/zeroing that slice
replaces/zeroes exactly that head's additive contribution to the residual
stream. monoT5 scoring uses a single decoder step, so every decoder head
activation has shape `(1, 1, 768)` regardless of input — control/attack/clean
activations are always shape-compatible (alignment only matters for building
the padded-control **encoder** input, reused from Experiment 1).

## Interventions per (head, example)

| column | base input | head replacement |
|---|---|---|
| `score_patched_fwd` | padded control | attack activation |
| `score_patched_rev` | attack | control activation |
| `score_ablated_zero` | attack (grids) / clean (side-effect) | zero vector |
| `score_ablated_mean` | attack (grids) / clean (side-effect) | padded-control activation |

Effects (same formulas as Experiment 1, Eq. 4–6, per head):
`fwd_effect = (fwd − control) / Δ`, `rev_effect = (attack − rev) / Δ`,
`combined = min(fwd, rev)`, with `Δ = attack − control`; examples with
`|Δ| < 1e-4` are skipped (same `SKIP_EPSILON`).

**Note:** on the attack base, *mean ablation* and *reverse patching* are the
same computation (attack run, control activation at that head), so in Grid A/B
`score_ablated_mean == score_patched_rev` by construction; it is computed once
and written to both columns. On the clean base (side-effect runs) mean
ablation is a distinct computation.

`score_drop_{zero,mean} = score_before − score_ablated_{zero,mean}`, where
`score_before = score_attack` in Grid A/B and `score_clean` in side-effect runs.

## The four runs

| run | attacks | examples | ops |
|---|---|---|---|
| `grid_a` | all 105 | 10/attack | fwd, rev, zero, mean (attack base) |
| `grid_b` | `relevant_start_5` | 100 | same as grid_a |
| `clean_a` | grid_a's matched clean inputs | 10/attack | zero, mean (clean base) |
| `clean_b` | grid_b's matched clean inputs | 100 | zero, mean (clean base) |

Examples are **reused from Experiment 1's selections**
(`../01_monot5_layer_patching/outputs/attacks/{attack}/scores/selected_examples.jsonl`, top-n by
`attack_delta_vs_control`), so head-level results are directly comparable to
the layer-level results. If a selection file is missing, the attack TSV is
scored from scratch with the same upstream logic
(`src.scoring.score_pairs_general` — added to `src/scoring.py` as the one
small upstream change of this experiment; scripts 00–11 are untouched).

## Performance design (exact, verified by tests)

1. **Encoder-output reuse** — all interventions are decoder-side, so the
   encoder runs once per input and its hidden states are reused for every
   patched decoder pass.
2. **Batched-over-heads decoder passes** — all 12 heads of one
   (layer, component) are scored in a single decoder pass of 12 identical
   rows, where row *h* patches only head *h* (block-diagonal mask).

## Usage

```bash
conda activate advseq2seq
cd patch_code/analysis/03_monot5_head_patching_ablation

bash bash/run_tests.sh          # unit tests (tiny random T5, CPU)
bash bash/run_smoke.sh          # end-to-end smoke: 2 attacks x 2 examples → outputs_smoke/
bash bash/run_all.sh            # full pipeline (resume-safe), or:
bash bash/run_all.sh configs/default.yaml --run grid_a   # one run at a time

python3 scripts/02_aggregate.py --config configs/default.yaml
python3 scripts/03_make_plots.py --config configs/default.yaml
```

SLURM: see the header of `bash/run_all.sh`.

## Outputs

```
outputs/
  {grid_a,grid_b,clean_a,clean_b}/attacks/{attack}/
    head_results.csv            # PRIMARY: one row per (head, example)
    selected_examples.jsonl     # exact examples used
    status.json                 # resume/status bookkeeping
  {run}/aggregated/
    head_attack_summary.csv     # derived: mean±std per (head, attack)
    head_summary.csv            # derived: mean±std per head (pooled)
  plots/
    head_heatmap_grid_a.png             # 1. 288-head localisation map
    zero_vs_mean_ablation_scatter.png   # 2. attack-specific vs generally-important
    per_head_effect_across_attacks.png  # 3. universal vs token-specific heads
    clean_vs_attack_drop.png            # 4. safe vs risky ablation targets
    head_heatmap_grid_b.png             # 5. Grid B sanity check vs Grid A
```

## Deliberately not implemented yet

Multi-head / joint ablation of an "important head set" — deferred until Grid
A/B results are reviewed and an importance threshold is picked.
