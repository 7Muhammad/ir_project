# Experiment 2 — Mean-Diff Direction Analysis + Intervention (Defense Test)

Builds on Experiment 1 (`01_monot5_layer_patching/`, layer-level activation
patching) and Experiment 3 (`03_monot5_head_patching_ablation/`, per-head
decoder patching/ablation). Computes a linear "mean-diff" direction —
`mean(attack_activations) - mean(control_activations)` — at attention heads,
then tests whether *subtracting* that direction from an attacked input's
activation pushes its score back toward the control baseline (a proto-defense),
and whether *adding* it to a clean input's activation is sufficient to push
the score toward attack-like values.

**Scope limitation — read this first:** the direction is fit and tested on
the same (in-sample) pool of examples. This establishes feasibility, not a
generalizing defense claim. See `DECISIONS.md` for this and every other
deliberate deviation from the original task spec (folder path, the exact
flagged-head list, why `decoder_self_attn` is in scope, the two-tier
fitting design, the encoder pooling rule).

## Research questions

- **Part 1 (descriptive):** Does a simple mean-diff direction exist per
  (layer, component[, head])? Does its norm predict Experiment 3's causal
  `combined_effect_mean` ranking?
- **Part 2a (causal, priority — defense):** Subtracting `scale * direction`
  from the attacked input's activation at Experiment-3-flagged heads — does
  any scale push the score back toward control without overcorrecting?
- **Part 2b (causal — sufficiency):** Adding `scale * direction` to the
  clean input's activation at the same heads — is the direction alone
  sufficient to move a clean score toward attack-like values?

## Formulas

```
direction = mean(attack_activations) - mean(control_activations)

2a: score_modified = score(attack_input, head_activation - scale * direction)
    delta_toward_control = |score_attack - score_control| - |score_modified - score_control|

2b: score_modified = score(clean_input, head_activation + scale * direction)
    delta_toward_attack = |score_clean - score_attack| - |score_modified - score_attack|
```

Positive `delta_toward_control` / `delta_toward_attack` = the intervention
moved the score in the intended direction.

## Scope

- **Locations (Part 1):** `encoder_self_attn`, `decoder_self_attn`,
  `decoder_cross_attn` (the third added beyond the spec's two — see
  DECISIONS.md item 3), both whole-vector (per layer) and per-head
  granularity.
- **Intervention scope (Part 2):** the 22 heads Experiment 3 flagged
  (`combined_effect_mean > 0.02` on its grid_a aggregate — 21
  `decoder_cross_attn` + 1 `decoder_self_attn`). No encoder heads (Exp3
  never scored any).
- **Tiers**, mirroring Experiment 3: `grid_a` (breadth — all 105 attacks, up
  to 10 examples each) and `grid_b` (depth — `relevant_start_5` only, up to
  100 examples). The direction-fitting pool and the intervention test pool
  are identical within each tier.
- **Scales:** {0.5, 1.0, 1.5}. **Interventions:** subtract (2a, defense,
  priority) and add (2b, sufficiency).

## Usage

```bash
conda activate advseq2seq

# Smoke test first (CPU, a few minutes):
bash bash/run_smoke.sh

# Full pipeline (interactive or via run_job.sh on SLURM — see bash/run_all.sh header):
bash bash/run_all.sh configs/default.yaml
```

Or step by step:

```bash
python scripts/00_derive_flagged_heads.py --config configs/default.yaml
python scripts/01_compute_directions.py   --config configs/default.yaml   # Part 1
python scripts/02_run_interventions.py    --config configs/default.yaml   # Part 2
python scripts/03_aggregate.py            --config configs/default.yaml
python scripts/04_make_plots.py           --config configs/default.yaml
```

`pytest tests/` runs the CPU-only unit tests (tiny random `T5Config`, no
downloads) — hook correctness (encoder per-head slicing, additive shift,
no-leaked-hooks), direction arithmetic, and the delta-metric formulas.

## Outputs

```
outputs/
  flagged_heads.json                              # script 00
  directions/
    {tier}_directions.pt                            # DirKey -> Tensor, script 01
    {tier}_direction_norms.csv
    direction_norm_vs_exp3_effect.csv               # script 03 cross-reference
  interventions/{tier}/attacks/{attack}/
    rows.csv                                        # per-example, script 02
    status.json
  interventions/{tier}/aggregated/                  # script 03
    defense_by_head_scale_attack.csv
    defense_by_head_scale.csv
    best_scale_by_head.csv
    sufficiency_by_head_scale_attack.csv
    sufficiency_by_head_scale.csv
  plots/                                            # script 04
    defense_by_scale_grid_a.png
    defense_across_attacks_best_scale_grid_a.png
    direction_norm_vs_exp3_effect.png
    sufficiency_by_scale_grid_a.png
```

## Deliberately not implemented yet

- Held-out attack generalization (fit direction on a subset of attacks, test
  on the rest) — the current design is explicitly in-sample only.
- Joint / multi-head interventions (shifting several flagged heads at once)
  — each head is intervened on independently.
- Encoder-head interventions — Part 1 computes `encoder_self_attn`
  directions descriptively, but Part 2 never intervenes there (Experiment 3
  flagged no encoder heads to scope it against).
