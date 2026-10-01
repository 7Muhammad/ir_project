# Experiment 19 — decisions

1. **Population = Exp 18 exactly.**
   - Instance order, metadata, scores, and success labels are read from the
     Exp 18 Parquet files. The attacked texts come from the same Exp 16
     stage-17 records.
   - The full run asserts that the counts equal Exp 18's `population.json`.
2. **Representation.**
   - The pre-o_proj head output is the input of `SelfAttention.o` sliced per
     head (the `EncoderHeadCapture` convention).
   - The hook reduces each layer immediately; no activation is kept.
3. **Regions.**
   - Only `query` and `orig_doc` are used, with Exp 18's exact masks.
   - Whole-sequence regions are out of scope here.
4. **Metrics and CKA.**
   - Same formulas as Exp 18: 1 − cos = ½‖x̂ − ŷ‖², and normalized L2 with a
     denominator of max(‖x‖, 1e-6).
   - Linear CKA is streamed per head as float64 64×64 Gram statistics per
     success × relevance cell. The query-centred version centres each
     condition on its own per-query mean.
5. **Causal set.**
   - The binary set is the Exp 13 sender list (18 heads). It is asserted equal
     to Exp 11 `combined_effect_mean > 0.02`.
   - The continuous comparison uses the Exp 11 effect for all 144 heads.
   - No list is hard-coded.
6. **Layer control.**
   - Change magnitude differs systematically by layer, and 16 of the 18 causal
     heads sit in L8–L11.
   - The primary causal test therefore uses WITHIN-LAYER percentiles: the mean
     percentile of the causal heads, against a layer-stratified permutation
     null that draws the same number of heads per layer (20,000 permutations).
   - Global top-k enrichment (hypergeometric) and global rank correlation are
     reported as descriptive only.
7. **Uncertainty.**
   - Head means get a query-level bootstrap (2,000 replicates).
   - The within-layer causal percentile gets a query-bootstrap CI (500
     replicates).
   - CKA is a descriptive population value.
   - p-values from 422k instances are never the main evidence.
8. **Success contrast.**
   - Per head: successful − unsuccessful mean change within each attack
     configuration (configurations with ≥ 5 of each), averaged over
     configurations and divided by the pooled SD.
   - This removes the configuration-composition confound.
9. **Correlation with Δscore.**
   - The main measure is the within-attack-configuration Spearman (ranks inside
     each attack, then Pearson).
   - Pooled and per-query ρ (queries with ≥ 10 instances) are also reported.
10. **Storage.**
    - The per-instance arrays and CKA statistics are deleted after stage 02
      (user decision, 2026-09-30).
    - Recomputing anything beyond the kept tables needs a rerun of stage 01
      (~30 min on an L40).
