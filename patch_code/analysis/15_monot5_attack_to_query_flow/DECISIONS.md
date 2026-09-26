# Experiment 15 — Decisions

Items 1–32 were fixed with the researcher before implementation. They are
**locked**: do not change them, and do not add the omitted controls ad hoc.
Items 33+ are implementation choices that could matter scientifically. Read
this file before editing any config value, sampling rule, position
definition, hook or metric.

## Scope

1. **Heads: the 18 canonical important encoder heads only.** Source:
   `13_monot5_encoder_decoder_path_patching/configs/heads/encoder_senders.json`
   (Exp 11 canonical run, combined effect > 0.02), loaded with Exp 13's
   `load_senders`, which asserts exactly 18 unique heads. Each head is
   tested on its own. Only `smoke.yaml` may set `heads.subset`.
2. **No scan of all 144 encoder heads.**
3. **No unimportant-head control set** (matched or random).
4. **No joint-head interventions.** Every patched pass edits exactly one
   head. Joint heads are left for a separate, future experiment.
5. **All 105 attacks** (7 tokens × {start, end, random} × reps 1–5). The
   grid comes from Exp 01's `configs/multi_attack.yaml` through
   `attacks.inherit_from`; it is never copied.

## Population

6. **Successful instances only:** `score_attack − score_control >
   SKIP_EPSILON`, with `SKIP_EPSILON` imported from `src.patching`
   (Exp 01, 1e-4). Exp 15 defines no epsilon of its own; the config key
   `success_epsilon_source` is documentation and is asserted at load time.
7. **At most 50 examples per attack.** If an attack has fewer than 50
   successful instances, all of them are used. In practice every attack has
   ≥ 80 (smallest pool: 80), so the full run has exactly 50 × 105 = 5,250
   examples.
8. **Seed 42.** Each attack's successful pool is shuffled with
   `random.Random(f"42:{attack}")` and the first 50 are taken. This is
   Exp 13's per-attack scheme, so Exp 13's first n examples are a prefix of
   ours (tested).
9. **One immutable sample** (`outputs/sample_manifest.jsonl`), shared by
   every head, condition, direction and the whole-head reference. Stage 00
   refuses to overwrite a different manifest without `--force`. Every later
   unit records the manifest's sha256 and fails loudly if it is resumed
   against a different manifest or head set.

## Positions

10. **All injected positions form ONE source set A.** A covers every
    repetition and every scattered span of a random attack. A comes from
    Exp 01's token-level alignment
    (`build_padded_control_and_attack_encodings_general`), never from a
    string search.
11. **Targets are exact T5/SentencePiece token positions.** There is no
    word reconstruction and no subword merging. Token strings are stored
    for interpretability only.
12. **Targets are query text only**, from Exp 06's
    `find_query_and_doc_spans`. `▁ Query :`, `▁Document :`, `▁Relevan t :`
    and `</s>` are excluded; tested on real prompts.

## Intervention

13. **The edit happens before `o_proj`**, on the per-head slice
    `[h·64, (h+1)·64)` of the encoder self-attention `.o` input (the
    Exp 03/11 convention). It is not applied after head mixing, and never
    to the residual stream.
14. **Only the source-specific P·V message is replaced:**
    `m(q) = Σ_{a∈A} P[q,a] V[a]`, using the post-softmax P and the per-head
    V taken from the same run. Keys, values, logits and probabilities are
    never transplanted on their own.
15. **No attention renormalization** and no −inf logits: the other sources'
    contributions are untouched.
16. **The padded control (Type B) is the matched donor and baseline.**
    m_control is computed explicitly, not hard-coded to zero. It comes out
    exactly 0 because masked keys get probability exactly 0 in fp32.
17. **Both directions are run.** fwd (sufficiency): receiver = control,
    `z ← z_ctrl − m_ctrl + m_atk`. rev (necessity): receiver = attack,
    `z ← z_atk − m_atk + m_ctrl`.
18. **`e_combined = min(e_fwd, e_rev)`**, with
    `e_fwd = (S_ctrl,patched − S_ctrl)/δ`, `e_rev = (S_atk − S_atk,patched)/δ`
    and `δ = S_atk − S_ctrl`. This matches Exp 01/03/11. **Nothing is
    clipped.**
19. **The all-query and single-query-token conditions both run on all 105
    attacks.** The all-query condition is its own intervention; it is never
    a sum or mean of single-token runs.
20. **Single-token results are descriptive only:** no p-values and no FDR.
21. **No Exp 12 token categories** (content/stopword, matched/unmatched).

## Reference and baselines

22. **The whole-head reference is recomputed on this manifest** using
    Exp 11's `cache_head_inputs` and `head_scores_for_layer`. Exp 11's own
    aggregates are never used as a denominator.
23. **The edge/whole-head ratio is secondary.** It is not a headline
    number and is not tested. It is computed only when
    `|whole_head_combined| > SKIP_EPSILON` (the project's existing
    numerical guard) and is never clipped. Exp 11's 0.02 importance
    threshold is shown next to it as a flag, not used as a filter.
24. **Cached Exp 01 `control_score` / `attack_score` are the baselines**
    for every metric. Fresh live scores come for free from the capture
    passes and are saved per row. They are checked against the cache in
    stage 01 (tolerance `fresh_vs_cached_score_atol = 5e-3`) and summarised
    over all 5,250 examples in `05_aggregate/baseline_consistency.json`.
    They are never used for normalisation.

## Aggregation and statistics

25. **Attacks get equal weight:** metrics are averaged within each attack,
    then the 105 attack means are averaged. This applies to e_fwd, e_rev,
    e_combined, raw_fwd and raw_rev.
26. **95% bootstrap CI** (percentile), 10,000 replicates.
27. **The bootstrap resamples examples within each attack**; attacks are
    held fixed. RNG: `numpy.default_rng(SeedSequence([42, head_index]))`.
28. **One-sided sign-flip test** on each head's 105 attack-level mean
    e_combined values. 100,000 flips, seed 42,
    `p = (extreme + 1)/(N + 1)`.
29. **Exactly 18 primary tests**, all-query only. There are no per-attack,
    per-direction or per-token tests.
30. **Benjamini–Hochberg at α = 0.05** over the 18 p-values. Every head is
    reported, significant or not.
31. **No representation-space metrics** (cosine, norms or probes as
    outcomes). The only outcome is the monoT5 score. Message norms and
    attention mass are saved as sanity diagnostics only.
32. **No ordinary-document source-position control**, and no random
    source or target controls. The matched padded-control message is the
    only control.

## Implementation choices (added during implementation)

33. **Alignment boundary non-uniqueness is kept as Exp 01 labels it.** In
    10 of 5,250 sampled examples, all of them `relevant_random_*`, an
    injected `▁relevant :` sits immediately after `▁Document :`. The token
    diff then labels the template `:` plus `▁relevant` as inserted instead
    of `▁relevant` plus the injected `:`. The token ids are identical, so
    both labelings are valid insertion sets. We keep Exp 01's labeling
    because the padded control masks exactly those positions: A must equal
    the masked set for m_control ≈ 0 and for consistency with the cached
    baselines. These examples are flagged (`alignment_boundary_shift`) in
    the manifest and in all raw outputs, and nothing is dropped. The query
    targets are never affected.
34. **Hook points were verified against the installed transformers 5.9
    source.** P is `output_attentions=True` (post-softmax, (1,H,T,T)). V is
    a forward hook on `SelfAttention.v`, viewed as (H,T,64) exactly as in
    `T5Attention.forward`. z is a pre-hook on `.o`. The decomposition
    `z_h(q) = Σ_j P V` is asserted on every example (atol 1e-4; observed
    ~0).
35. **Precomputed receiver messages are exact, not an approximation.** The
    edit is at layer L's `.o` input, after that layer's P and V are
    computed, so the receiver's layer-L message in the patched pass equals
    the one from its unpatched capture pass. Check N confirms this against
    `exp15lib/reference.py`, which recomputes P and V live inside the
    patched pass from the module's q/k/v weights, with explicit loops and
    batch size 1.
36. **Optimisation: batching.** Control and attack have identical length,
    so interventions (heads × directions × targets) for one example are
    stacked as independent batch rows (≤ `max_rows_per_batch`). This is
    exact up to float noise; batch-size invariance is tested. The encoder
    is re-run in full for every row, with no layer-prefix reuse, which
    keeps the code simple at a modest compute cost.
37. **Optimisation: the whole-head subset mask.** Exp 11's
    `head_scores_for_layer` sizes its batch from the head mask it is given.
    We pass only the canonical heads' rows of Exp 11's block-diagonal mask,
    which removes 70% of whole-head rows. Equality with the full 12-row
    pass is tested.
38. **Check M (whole-head consistency):** edge patching with
    A = Q = all positions is algebraically identical to whole-head
    patching. Unit tests and stage 01 assert this against Exp 11's code.
39. **Resume unit = one attack**, covering all heads, both directions and
    all targets. `--attack-start/--attack-end` split the grid across SLURM
    jobs deterministically. Stages 05–07 refuse to run on an incomplete
    grid.
40. **The all-query stage and the whole-head stage run before the
    single-token stage** in `run_all.sh`. The primary results therefore
    finish first if the largest (descriptive) stage runs out of time.
41. **Full fp32 is used** (`allow_tf32: false`). Tolerances assume fp32.
42. **Post-hoc descriptive robustness (stage 08), added after seeing the
    full results.** δ is often tiny in the successful-instance population
    (median ≈ 0.07 logits; about 10% of examples below 0.01), so equal-weight
    means of per-example ratios can be dominated by near-zero-δ examples.
    `scripts/08_posthoc_robustness.py` therefore reports four alternative
    summaries:
    - ratio-of-means per attack;
    - the median of per-example e_combined;
    - the mean over examples with δ ≥ 0.1;
    - the fraction of attacks with a positive raw effect.

    These are **descriptive only**. The headline metric, the aggregation,
    the 18 tests and BH (items 18, 25–30) are unchanged and remain the
    primary analysis.
