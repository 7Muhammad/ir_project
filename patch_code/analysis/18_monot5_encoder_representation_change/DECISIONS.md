# Experiment 18 — decisions

1. **Own experiment folder, Exp 16 inputs read-only.**
   - The population, alignment rule, and success definition are exactly those
     of Exp 16 stages 17/18; nothing is re-derived.
   - Exp 16 outputs are never written.
   - The code was first prototyped as Exp 16 stages 25/26 and moved here at the
     user's request (2026-09-30). The full run (SLURM 1428822) was launched from
     that prototype. Its outputs were moved to `outputs/01_repr_change/`
     unchanged; the ported code reproduces the prototype smoke outputs exactly
     (max diff 0).
2. **States.**
   - The main layer states are the encoder block outputs L00–L11. L11 is taken
     BEFORE the final norm, matching Exp 16 `L11_post_mlp`.
   - `embedding` and `final_norm` are separate reference rows.
3. **Regions.**
   - `orig_doc` is defined through the clean-prompt mapping: *P* =
     non-inserted positions, and the passage is `P[d0c:d1c]`. It therefore
     holds exactly the passage tokens.
   - In the 1,270 Exp 16 DECISIONS-40 boundary-shift instances, `orig_doc`
     excludes the template `:` that the Exp 16 control doc mask contains.
   - The control's full-visible pool equals its `whole_shared` pool.
   - Full-visible regions are never compared token-wise.
4. **Numerics.**
   - The model runs in fp32 without TF32: the cached scores are fp32, and T5 is
     unsafe in fp16. bf16 would blur 1 − cos values of order 1e-4.
   - Control and attack of a pair run in the SAME batch.
   - 1 − cos is computed as ½‖x/|x| − y/|y|‖² to avoid cancellation.
   - Normalized L2 uses a denominator of max(‖x_control‖, 1e-6); the count of
     near-zero norms is reported.
   - Invariant: at the embedding state, the shared regions must show exactly
     zero change (checked on every instance).
5. **Linear CKA.** The formula is CKA = ‖Ycᵀ Xc‖²_F / (‖Xcᵀ Xc‖_F ‖Ycᵀ Yc‖_F),
   with rows = instances (768-d pooled states).
   - **Accumulation:** streamed as float64 Gram matrices plus per-query sums per
     base cell (success × relevance). Every group is exact by summation.
   - **Query-centred CKA:** control and attack are each centred on their own
     per-query mean within the analysed population. A query with one instance
     contributes zero and is reported (`n_queries_singleton`).
   - CKA is never correlated with per-instance Δscore.
6. **Uncertainty and correlation.**
   - **Means:** instance-weighted, with a query-level bootstrap (2,000
     replicates, seed 42, 95% percentile CI).
   - **Spearman ρ(change, Δscore):** change is oriented so that larger means
     more change. Three variants are reported:
     - pooled ρ, whose p-value ignores clustering;
     - within-attack-configuration rank correlation;
     - per-query ρ (queries with at least 10 instances).
7. **Storage.** Only per-instance scalar metrics are persisted; see the README.
