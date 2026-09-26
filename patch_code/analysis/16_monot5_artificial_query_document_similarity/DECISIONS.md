# Experiment 16 — Decisions

Items 1–37 were fixed with the researcher before implementation. They are
**locked**. Do not change them after seeing results. Items 38+ are
implementation choices that could matter scientifically. Read this file
before you edit any config value, population rule, mask, hook or statistic.

## Scope

1. **Exp 16 is similarity-only.** It does no patching, ablation, steering or
   head intervention, and it does not depend on Exp 15 outputs.

## Metric

2. **Query and document vectors are mean-pooled separately** at each
   checkpoint: `Q_c = mean(query-text states)`, `D_c = mean(document states)`.
3. **The primary metric is cosine similarity** `cos(Q_c, D_c)`. There is no
   extra LayerNorm, no max/top-k token cosine and no token-pair matrix. Decoder
   and template states are not used.
4. **The query pool is the query text only.** It is `range(*query_span)` from
   Exp 06's `find_query_and_doc_spans`. `▁ Query :`, `▁Document :`,
   `▁Relevan t :` and `</s>` are excluded (tested on real prompts).
5. **The document pool is the document text only**, `range(*doc_span)`, for
   clean and qrel inputs.
6. **The attacked document pool includes the injected tokens.** It is
   `range(*doc_span_control_attack)`, which covers the passage plus every
   inserted token, all active.
7. **Padded-control insertion slots are excluded.** The control document pool
   is the doc span AND `attention_mask == 1`. Pad states are never averaged in
   to equalise token counts.

## Clean association (RQ1)

8. **The 500 canonical clean Type-A pairs establish the similarity↔score
   association.** Padded controls are not used here.
9. **Spearman is the primary clean association.** Pearson and BH over the 25
   Spearman p-values are stored as secondary columns.

## Genuine relevance (RQ2)

10. **Human TREC DL19 passage qrels are the genuine-relevance reference.**
11. **qrel 2 or 3 means strongly relevant.**
12. **qrel 0 means non-relevant.**
13. **qrel 1 is excluded.**
14. **Relevant and non-relevant documents are compared within query.**
15. **Classes are balanced per query with `n_q = min(|R_q|, |N_q|)`.** The
    smaller class is kept whole and the larger class is sampled.
16. **Seed 42 controls qrel sampling:** `random.Random(f"42:{qid}")` over the
    docids sorted as strings, so each query's sample is order-independent.
    It is drawn once and saved; it is never resampled per checkpoint.
17. **Human qrels alone define relevance.** monoT5 scores are never used to
    select or label documents. The fresh monoT5 score of each judged document
    is stored for description only.
18. **Queries get equal top-level weight:** documents are averaged within
    (query, class) first, then queries are averaged.

## Attack population (RQ3)

19. **All available examples are used.** The population is every pair in
    Exp 01's `outputs/attacks/{attack}/scores/all_scores.csv`, which is every
    canonical pair whose Exp 01 token alignment succeeds (stage 00 verifies
    this identity for each attack).
20. **There is no success filter.** `selected_examples.jsonl` (filtered on
    `attack − control > 0`) is not used. Pairs with positive, zero and
    negative `delta_score` all remain.
21. **All 105 attacks are used**, inherited from Exp 01's `multi_attack.yaml`.
22. **Attacks get equal top-level weight:** examples are averaged within each
    attack first, then the 105 attack means are averaged.

## Checkpoints

23. **There are exactly 25 primary encoder checkpoints:** `embedding`, then
    `L{00..11}_post_attn` and `L{00..11}_post_mlp`.
24. **The embedding checkpoint is included.** It is the tensor entering encoder
    layer 0: a forward-pre-hook on `encoder.block[0]`, equal to
    `embed_tokens(ids)` with eval-mode dropout as identity.
25. **Each layer is measured post-attention:** `encoder.block[L].layer[0]`
    output[0], which is `x + SelfAttention(LN(x))`.
26. **Each layer is measured post-MLP:** `encoder.block[L].layer[-1]` output,
    which is `h + FF(LN(h))`, the block output.
27. **Primary states are residual hidden states after the sublayers.** They are
    not attention probabilities, not the raw attention output, not Q/K/V and
    not per-head slices.
28. **The final encoder stack norm is not a 26th primary checkpoint.**
    `L11_post_mlp` is taken before `final_layer_norm`.

## Quantities and statistics

29. **`delta_sim(c) = sim_attack(c) − sim_control(c)`.**
30. **`delta_step(c) = delta_sim(c) − delta_sim(c−1)`**
    `= step_attack(c) − step_control(c)`. It is descriptive localisation, not
    an independent causal effect. It is undefined at the embedding checkpoint.
31. **`delta_score = score_attack − score_control`**, using the cached Exp 01
    scores.
32. **delta_sim↔delta_score is Spearman across the 105 attack-level means.**
    The pooled example-level Spearman is stored as a clearly labelled
    secondary diagnostic.
33. **Attack `delta_sim` is tested checkpoint by checkpoint**, with the 105
    attack means as the unit.
34. **The test is a one-sided sign flip** (H1: mean > 0). It uses 100,000
    Monte Carlo flips, `p = (#{null ≥ obs} + 1)/(N + 1)`, and a numpy
    `SeedSequence([42, checkpoint_index])`. This follows the same convention
    as Exp 15, implemented in `exp16lib/stats.py` because Exp 15 is not
    committed.
35. **BH-FDR is applied across the 25 attack checkpoint tests** at α = 0.05.
36. **The genuine-relevance gap is compared directly with the attack gap.**
    Both `delta_sim` and `delta_step` trajectories are compared on the same
    checkpoint axis. The ratio attack/real is only a secondary descriptive
    column.
37. **No causal conclusion is drawn from similarity alone.**

## Implementation choices (38+)

38. **Qrel sources.** The qrels come from ir_datasets
    `msmarco-passage/trec-dl-2019/judged`, the same id the upstream ECIR-24
    evaluation uses (`advseq2seq/retrieval_effectiveness/evaluation_utils.py`).
    ir_datasets downloads it from
    `https://trec.nist.gov/data/deep/2019qrels-pass.txt`. The passage text
    comes from ir_datasets `msmarco-passage` (the full MS MARCO v1 passage
    collection). Before this experiment the repo had no local qrels file.
    The ECIR-24 `bm25_19.tsv.gz` has text only for BM25 top-1000
    documents. Using it would have cut the balanced sample from 2,256 to
    1,170 docs per class and biased the relevant class toward
    lexically-matched documents. That file is used only to cross-check that
    passage and query text are identical where both exist; the check is
    recorded in `provenance.json`. Query text comes from the same ir_datasets
    id.
39. **No silent truncation.** Any prompt longer than `max_length = 512` raises
    an error. For the attack and clean populations, none occur (verified in
    stage 00). For qrel documents, a prompt that would not fit is removed
    from the candidate pool **before** balancing, and is counted and listed
    in `provenance.json`.
40. **Alignment boundary shift (the benign Exp 15 item 33 case).** When
    Exp 01's token diff labels the template `:` after `Document` as inserted,
    instead of an identical injected `:`, the Exp 01 control masks that
    template position. Exp 16 keeps Exp 01's control as it is, because the
    cached scores were computed on it. The attacked document pool is
    span-based and always holds exactly passage + injected tokens. The
    control pool then holds the passage plus one `:`. Such examples are
    flagged (`alignment_boundary_shift`) and counted in `provenance.json`.
41. **The attack manifest follows the canonical pair order.** Smoke may cap
    examples per attack by taking the first k in that order; the full run
    never caps.
42. **Scores.** Clean and attack/control scores are the cached Exp 01 values.
    The full run never rescores them. The smoke run rescores 3 clean pairs
    and 3 examples per attack with Exp 01's own functions and requires
    agreement within 5e-3 (the Exp 15 tolerance). Qrel documents have no
    cache, so their monoT5 score comes from the same forward pass (one
    decoder step).
43. **Numerics.** Everything runs in fp32 with TF32 disabled. Pooled vectors
    are accumulated in fp32 and the cosine is taken in float64. Attack and
    control sequences are batched, sorted by length and right-padded. Stage
    01 requires batched and single-sequence similarities to agree within
    1e-5, and it checks every hook's semantics on real monoT5 before any
    results are produced.
44. **Uncertainty bands** in the figures are ±1 SE across the top-level unit
    (queries or attacks). They are descriptive, not an extra test family.
