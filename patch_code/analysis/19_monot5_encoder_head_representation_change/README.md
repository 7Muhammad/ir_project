# Experiment 19 — Head-level encoder representation change (padded control → attacked input)

**Question:** which of the 144 encoder self-attention heads show the strongest
control → attack change? Do those heads overlap with the heads that earlier
patching found to be causally important? This follows up Exp 18, which looked
at the full residual stream.

**Representation:** the *pre-o_proj head-output representation*. For head *h*
of layer *l* at token *t*, it is H[l,h,t] ∈ R^64: the input of
`encoder.block[l].layer[0].SelfAttention.o`, sliced `[h*64, (h+1)*64)`. It is
the same tensor that `exp16lib.heads.EncoderHeadCapture` reads. It is **not**
the full encoder representation.

The analysis is representational, not causal. The Exp 11 / Exp 13 patching
results remain the causal evidence.

## Population

- **Instances:** exactly Exp 18's. Instance order, metadata, scores, and success
  labels come from `18_…/outputs/01_repr_change`; attacked texts come from the
  Exp 16 stage-17 manifest.
- **Counts:** 105 attacks, 422,206 instances, 43 queries, 4,022 pairs.
  - 131,314 successful, 290,892 unsuccessful.
  - qrel 2/3 = relevant, qrel 0 = non-relevant; qrel 1 excluded.
  - The 104 alignment failures are excluded.
- **Check:** stage 01 asserts that the counts equal Exp 18's.

## Regions

The masks are exactly Exp 18's (`exp18lib.repr_change.pair_regions`):

- `query`: the query text tokens.
- `orig_doc`: the clean passage tokens only. Inserted tokens and padding are
  excluded.

## Metrics

Computed per instance × layer × head × region:

- 1 − cos and normalized L2 of the mean-pooled 64-d vectors.
- Token-wise mean 1 − cos and mean normalized L2 over aligned tokens.

Population-level linear CKA is computed raw and query-centred, with Exp 18's
estimator and centring convention. It is exact, from float64 64×64 Gram
statistics streamed during the forward pass.

## Causal heads (read-only sources of truth)

- **Binary:** `exp13lib.head_lists.load_senders()`, the 18 Exp 11 heads whose
  canonical combined effect is > 0.02. Stage 02 asserts that this list equals the
  Exp 11 table thresholded at 0.02.
- **Continuous:** Exp 11 `outputs/head_summary_canonical.csv`,
  `combined_effect_mean` for all 144 heads.

## Validation

- Token IDs, masks, and inserted-token exclusion are checked on every pair
  (Exp 18 checks).
- Scores equal Exp 18's.
- On the first batch of every attack, `EncoderHeadCapture` runs on the same
  forward pass:
  - its o_proj input equals ours bitwise;
  - its 144 head query/document cosines equal the Exp 16 stage-18 all_layers
    cache.
- Identical control/attack inputs give 1 − cos = 0, L2 = 0, and CKA = 1.

## Storage

The user chose Option 1, then decided to keep only the summaries:

- Stage 01 writes per-instance metrics (`metrics/*.npy`, 1.95 GB) and CKA
  statistics (`cka_stats.npz`, 0.16 GB). Stage 02 needs both.
- Both are deleted after stage 02. The kept outputs are about 20 MB of tables
  and figures.
- `02_analyze_head_change.py --plots-only` redraws the figures from the kept
  CSVs.

## Run

```
bash bash/run_tests.sh      # unit tests (CPU)
bash bash/run_smoke.sh      # 4 attacks x 24 instances, CPU
# full (from /home/ghoummaid/IR): stage 01 on an L40 (~30 min), then stage 02 (CPU)
bash run_job.sh --job-name monot5_exp19_heads --gpu-type L40 --gpu-count 1 --cores 8 --time 08:00:00 \
  --output-dir ./patch_code/analysis/19_monot5_encoder_head_representation_change/slurm_logs \
  --command "cd /home/ghoummaid/IR/patch_code/analysis/19_monot5_encoder_head_representation_change && bash bash/run_all.sh"
```

## Outputs

- `outputs/01_head_change/`:
  - `cka.csv`: per layer × head × region × success group × relevance group.
  - `instances.parquet`, `population.json`, `checks.json`.
- `outputs/02_analysis/`:
  - `head_summary.csv`: mean, query-bootstrap CI, and median per head × region
    × metric × group.
  - `head_correlations.csv`: Spearman with Δscore, within attack configuration,
    pooled, and per query.
  - `causal_comparison.csv`, `causal_head_ranks.csv`, `topk_heads.csv`.
  - `layer_concentration.csv` (L6–L11), `success_contrast.csv`,
    `relevance_contrast.csv`.
- `outputs/02_analysis/plots/`:
  - `fig1`: 1 − cos heatmaps.
  - `fig2`: normalized L2 heatmaps.
  - `fig3`: 1 − CKA raw; `fig3b`: 1 − CKA query-centred.
  - `fig4`: causal heads vs other heads.
  - `fig5`: top-20 heads.
  - `fig6`: the L6–L11 transition.
  - `fig7`: success contrast and correlation with Δscore.
  - `fig8`: linear CKA by layer (head median/IQR/range vs Exp 18 residual stream); `fig8b`: the same for every head on the x-axis.

Report: `19_monot5_encoder_head_representation_change_report.md`.

See `DECISIONS.md`.
