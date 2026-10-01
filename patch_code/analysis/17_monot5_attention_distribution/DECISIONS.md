# Experiment 17 — decisions

1. **Sample.** Exactly the Exp 16 stage-22 sample, read through `exp16lib.token_sample.TokenSample`.
   The sample is never reconstructed. Stage 01 asserts the counts (6,325 / 253 / 43 / 1,026). It
   also checks that the recomputed monoT5 score equals the stage-22 score for every sequence, and
   that the pre-o_proj head outputs equal `heads.npy` for the first batches.
2. **Attention source.** Eager T5 attention from `model(..., output_attentions=True).encoder_attentions`,
   which is post-softmax with the additive mask already applied. Stage 01 verifies that masked keys
   (control insertion slots and batch padding) get exactly 0 weight. Metrics still restrict and
   renormalise explicitly over the chosen key mask, so they never depend on that behaviour.
3. **Shared-key scope.** For the attacked sequence, `attention_mask & ~inserted_mask`. For the
   control this equals its full-visible set. Paired deltas are therefore
   `attack(scope) − control(full_visible)`.
4. **Control region for `orig`.** The control's document (`doc_mask`), which is identical to the
   attack's original-document tokens. Stage 01 validates this pairing position by position.
5. **Distance.** Absolute token positions. Control and attack positions are aligned because the
   control keeps masked slots, which matches how T5's relative-position bias sees the sequence.
   Normalised by `n_tokens − 1` (unpadded length). Interpret descriptively only.
6. **top-1 is `max_attn`.** It is stored once.
7. **Normalised region mass.** `observed / (|target| / N_visible)`, computed over full-visible keys.
   The median over heads is well below 1 because template tokens (`</s>`, `:`) absorb much of the
   attention. The query→orig and orig→query rows are included as a size-normalised reference
   from the same sequence.
8. **Length.** Relevant documents are longer, so the number of visible keys differs between groups.
   Relevance summaries report Cohen d both raw and after regressing the metric on
   `n_keys_full_visible`, plus a Spearman correlation with it.
9. **Generic insertion effects.** Each paired summary also reports the same delta for unsuccessful
   attacks. A delta that is equally present in unsuccessful attacks is an insertion effect, not a
   success effect.
10. **Causal comparison.** Uses Exp 11 per-head combined effects (canonical `relevant_start_5`, and
    the sweep mean over the 12 stage-22 attacks and over all 105), read-only. It is not blocking.
11. **Head-index structure (not a bug).** monoT5 computes the relative-position bias only in encoder
    layer 0 and reuses it in every layer. Head h therefore has the same positional prior in all 12
    layers. Locality, direction and injected-mass heatmaps show column stripes: heads 7/8 barely
    attend from the query forward into the document, and heads 9/10 barely attend backward to the
    query. The same stripes appear for the query↔original-document reference, so they are
    positional, not attack-related. The injected-mass heatmap colour scale is clipped at log2 = ±3.
