# Experiment 16 — Artificial query-document similarity

**Hypothesis.** Keyword stuffing makes monoT5's internal query and document
representations look artificially alike. Genuinely relevant documents may
show a similar query–document similarity pattern. Exp 16 measures this; it
does **not** intervene.

This experiment is **observational and non-causal.** It has no activation
patching, head ablation, steering or edge patching, and it does not depend on
Exp 15. A positive, relevance-like, score-tracking similarity gap would at
most show that artificial query-document similarity is a *representational
signature* of the attack. Testing mediation needs a later intervention
experiment.

## Research questions

| RQ | Question | Population | Unit |
|----|----------|------------|------|
| RQ1 | On clean inputs, does higher query–document similarity go with a higher monoT5 score? | 500 canonical Type-A pairs | example (Spearman) |
| RQ2 | Are human-judged qrel 2/3 documents more query-similar than qrel 0 documents? | balanced within-query DL19 qrel sample | query (equal weight) |
| RQ3 | Does keyword stuffing raise similarity relative to the matched padded control? | all scored examples × all 105 attacks (**no success filter**) | attack (equal weight) |
| RQ4 | Do attacks that raise similarity more also raise the score more? | 105 attack-level means | attack (Spearman) |

A localisation question runs through all four: where in the encoder does
similarity appear, grow or diverge? The central comparison asks whether
adversarial injection (padded control → attacked) moves the query–document
relationship in the same direction and at the same depths as genuine
relevance (qrel 0 → qrel 2/3).

## Metric

At every checkpoint `c`:

```
Q_c = mean over query-TEXT positions of h_c
D_c = mean over document positions of h_c
sim_c = cos(Q_c, D_c)
```

The metric is taken on the model's actual residual states. There is no extra
LayerNorm, no max or top-k token cosine and no token-pair matrix. Decoder and
template states are not used.

**Query pool.** Only the query text, taken from Exp 06's
`find_query_and_doc_spans` (exact token-prefix probes, not string matching).
`▁ Query :`, `▁Document :`, `▁Relevan t :` and `</s>` are excluded.

**Document pool.**

| Input | Pooled positions |
|-------|-----------------|
| clean / qrel | passage tokens only |
| attacked | passage tokens **plus every injected attack token** (all active) |
| padded control | active document positions only; the masked insertion slots are **excluded** |

Attack tokens are included on purpose. The question is whether adding them
makes the document representation *as the model sees it* more query-like.
The control's insertion slots are pad tokens with `attention_mask = 0`. They
carry no content, and averaging them in would dilute `D` with states nothing
attends to. Batch padding never enters either pool.

## The 25 encoder checkpoints

The hook points come from the installed Transformers 5.9.0 source,
`models/t5/modeling_t5.py`:

| Name | Hook | Meaning |
|------|------|---------|
| `embedding` | forward-pre-hook on `encoder.block[0]`, arg 0 | `dropout(embed_tokens(ids))`, the exact tensor entering layer 0 (dropout is identity in eval; T5 does not scale embeddings) |
| `L{L}_post_attn` | forward hook on `encoder.block[L].layer[0]` (`T5LayerSelfAttention`), `output[0]` | `x + SelfAttention(LN(x))`: the residual stream after the attention update, before the FFN |
| `L{L}_post_mlp` | forward hook on `encoder.block[L].layer[-1]` (`T5LayerFF`), output | `h + DenseReluDense(LN(h))`: the block output passed to layer L+1. `L11_post_mlp` is taken **before** `encoder.final_layer_norm` |

That gives `1 + 12 × 2 = 25` checkpoints. The final-normalised encoder output
is **not** a 26th checkpoint. Every state has shape `[batch, seq_len, 768]`.

A single forward pass yields all 25 similarities. Each hook reduces its state
to one cosine per sequence straight away, so no hidden-state tensor is stored.

Stage 01 checks the semantics on real monoT5 before producing any result
(`exp16lib/sanity.py`; the unit tests repeat the checks on a tiny T5):
- `embedding == embed_tokens(ids) == hidden_states[0]`
- `post_attn[L] == block_in[L] + SelfAttention(...)[0]`, and `post_attn[L]` differs from the raw attention output
- `post_mlp[L] == post_attn[L] + FF(LN(post_attn[L]))`
- `post_mlp[L] == block_in[L+1]`
- `final_layer_norm(post_mlp[11]) == last_hidden_state`
- batched and single-sequence similarities agree within 1e-5

The embedding checkpoint separates similarity that exists as soon as the
tokens are inserted from similarity the encoder builds afterwards.

## Analyses

**A. Clean similarity vs monoT5 score.** For each checkpoint, Spearman
`ρ(sim_clean(c), score_clean)` over the 500 canonical pairs. `score_clean` is
Exp 01's cached Type-A `original_score`. Pearson and BH-adjusted Spearman
p-values are secondary columns.

**B. Genuine relevance (TREC DL19 qrels).**
- **Groups:** qrel 2/3 = relevant, qrel 0 = non-relevant, qrel 1 excluded.
- **Balancing:** within each query, `n_q = min(|R_q|, |N_q|)`. The smaller
  class is kept whole and the larger class is sampled with
  `random.Random(f"42:{qid}")`. Only queries with both classes are kept.
- **Aggregation:** `sim_rel_q(c)`, `sim_nonrel_q(c)` and
  `delta_sim_real_q(c) = sim_rel_q − sim_nonrel_q` are computed per query,
  then averaged over queries with **equal query weight**. The step
  `delta_step_real(c) = delta_sim_real(c) − delta_sim_real(c−1)` is also
  computed. This analysis is descriptive; per-query gaps are saved for later
  tests.
- **Labels:** only human judgments define the classes. monoT5 scores of the
  judged documents are stored for description only.

**C. Attacked vs padded control.** Covers every scored example of every
attack, with no success filter: rows with negative or zero `delta_score`
stay.
- **Per example:** `delta_sim(c) = sim_attack(c) − sim_control(c)` and
  `delta_score = score_attack − score_control` (cached Exp 01 scores).
- **Aggregation:** mean within each attack, then **equal weight across the 105
  attacks**.
- **Accumulation:** `step_attack`, `step_control` and `delta_step(c) =
  delta_sim(c) − delta_sim(c−1)`. This is descriptive localisation, not a
  causal effect.
- **Test:** one-sided sign flip (H1: mean `delta_sim(c)` > 0) on the 105
  attack means, with 100,000 flips, `p = (extreme+1)/(N+1)` and seed 42.
  **BH-FDR** is applied across the 25 checkpoints at α = 0.05.

**D. Similarity increase vs score increase.** For each checkpoint,
`ρ_delta(c)` is Spearman across the 105 attacks of `mean_delta_sim_k(c)`
against `mean_delta_score_k` (primary). A pooled example-level Spearman is
stored as a **secondary** diagnostic.

**E. Genuine vs adversarial.** `delta_sim_real(c)` and `delta_sim_attack(c)`
are compared on one axis, along with `delta_step_real` and
`delta_step_attack`. The comparison asks whether both gaps are positive,
whether they emerge at similar depths, whether they grow through the same
sublayer types, and how large the attack shift is relative to the genuine
one. The attack/real ratio is only a secondary descriptive column. The two
trajectories come from different populations and units, so a match is not
causal identity.

**Optional breakdowns (descriptive).** Equal-weight `delta_sim` by attack
token, insertion position and repetition count. No new test family is added.

## Data sources

The exact sources, with sha256 hashes, are written to
`outputs/manifests/provenance.json`.

| What | Source |
|------|--------|
| Canonical 500 Type-A pairs | `01_monot5_layer_patching/outputs/pairs/pairs.jsonl` |
| Clean score | `01_monot5_layer_patching/outputs/scores/all_scores.csv` (`original_score`) |
| Attack grid | `01_monot5_layer_patching/configs/multi_attack.yaml` via `attacks.inherit_from` (Exp 01 `src.attack_registry`) |
| Full attack population + scores | `01_monot5_layer_patching/outputs/attacks/{attack}/scores/all_scores.csv` (all aligned pairs) plus texts from `.../pairs/pairs.jsonl`. **`selected_examples.jsonl` is success-filtered and not used.** |
| Padded control / alignment | Exp 01 `build_padded_control_and_attack_encodings_general` (the construction behind the cached scores) |
| DL19 qrels | ir_datasets `msmarco-passage/trec-dl-2019/judged` (upstream ECIR-24 uses the same id), downloaded from `https://trec.nist.gov/data/deep/2019qrels-pass.txt` into `~/.ir_datasets/msmarco-passage/trec-dl-2019/qrels` |
| Qrel passage and query text | ir_datasets `msmarco-passage` (full MS MARCO v1 passage collection); cross-checked against the ECIR-24 `data/bm25_19.tsv.gz` where both exist |

Stage 00 checks the population and fails loudly on any mismatch:
- every attack's pairs equal the canonical 500 pairs;
- the pairs whose Exp 01 alignment fails are exactly the pairs missing from
  `all_scores.csv`, so the scored set is the complete aligned population;
- every example encodes without truncation, with non-empty query and document
  pools.

## Pipeline

```
00_prepare_manifests  -> outputs/manifests/{clean.jsonl, attacks.jsonl.gz, qrel_balanced.jsonl,
                          qrel_query_summary.csv, attack_population_summary.csv, provenance.json}
01_run_clean_similarity -> 01_clean/{clean_similarity.csv, sanity_report.json}
02_run_qrel_similarity  -> 02_qrel/per_query/{qid}/rows.csv.gz, 02_qrel/qrel_similarity.csv.gz
03_run_attack_similarity -> 03_attack/per_attack/{attack}/rows.csv.gz, 03_attack/attack_similarity.csv.gz
04_analyze              -> 04_analysis/*.csv, summary.json
05_plot                 -> plots/fig1..fig9, figS1..S2
06_plot_successful_vs_genuine_similarity -> plots/fig_successful_attacks_vs_genuine_similarity.png (+ CSV in 04_analysis)
07_run_decoder_message_similarity        -> 07_decoder/{qrel_decoder,attack_decoder}.csv.gz
08_plot_decoder_message_similarity       -> plots/fig_decoder_message_similarity_four_populations.png (+ CSV)
09_run_head_similarity                   -> 09_heads/{qrel_heads,attack_heads}.csv.gz
10_plot_head_similarity                  -> 10_head_analysis/*.csv, plots/fig_heads_*.png
11_run_decoder_probe_similarity          -> 11_decoder_probe/{qrel,attack}_decoder_probe.csv.gz
12_plot_decoder_probe_similarity         -> plots/fig_decoder_probe_similarity_four_populations.png (+ CSV)
13_run_late_encoder_heads                -> 13_late_encoder_heads/{qrel,attack}_late_encoder_heads.csv.gz
14_plot_late_encoder_heads               -> 10_head_analysis/late_encoder_heads_L9_L11.csv, plots/fig_heads_encoder_all_L9_L11_four_populations.png
15_head_anomaly_detection                -> 15_head_anomaly/{anomaly_predictions.csv.gz, per_head_abnormality.csv, detector_threshold_curve.csv, fold_summary.csv, overall_summary.json}, plots/fig_anomaly_*.png
16_head_anomaly_attribution              -> 15_head_anomaly/{per_head_attack_anomaly_analysis.csv, topk_head_detector_auroc.csv, attribution_summary.json}, plots/fig_anomaly_attr_*.png
17-21 paired extension                   -> see 'Paired extension (stages 17-21)' below
```

### Supplementary level analyses (stages 06–16)

These stages plot absolute similarity levels, not gaps, for four populations:
clean qrel 2/3 (genuinely relevant), clean qrel 0 (non-relevant), and
attacked inputs split by example-level `delta_score > 0` (successful) or
`≤ 0` (unsuccessful).
- Qrel groups are weighted equally per query.
- Attack groups are averaged within each attack first, then weighted
  equally across attacks.
- Bands are ±1 SE across queries or across attack configurations.
- The success split exists only in these descriptive plots; the primary
  analyses stay unfiltered.

| Stage | Quantity |
|-------|----------|
| 06 | the encoder `sim(c)` of this README, at the 25 encoder checkpoints |
| 07/08 | per decoder layer: cosine between the cross-attention message read from query-text positions and the message read from document positions, `m_S = o(Σ_{j∈S} P·V)` (`exp16lib/decoder.py`). The query + document + template parts must reproduce the cross-attention output (relative error ≤ 1e-5, checked every batch). |
| 09/10 | the same quantities at the canonical important heads (Exp 13 lists: 18 encoder heads, 31 decoder cross-attention heads). Encoder: pre-`o_proj` 64-d head slice, pooled over query vs document. Decoder: per-head query-sourced vs document-sourced 64-d contribution (`exp16lib/heads.py`). |
| 11/12 | decoder residual-stream probe: the encoder runs once, then the first decoder step runs twice with cross-attention restricted to query-text positions or to document positions. `cos(h_query-only, h_doc-only)` is taken at 37 decoder checkpoints: embedding, then post self-attention, post cross-attention and post MLP for each layer, with L11 taken before `final_layer_norm` (`exp16lib/decoder_probe.py`). Restricting cross-attention is a probe, not the model's normal run. The two pre-cross-attention checkpoints must be exactly 1. |
| 13/14 | the stage 09 encoder-head metric for **all 36** self-attention heads of encoder layers 9–11 (`L9H0`–`L11H11`), from an encoder-only pass. The 16 heads shared with stage 09 must reproduce its cache (`head_crosscheck_atol`). The CSV adds `success_gap`, `genuine_gap`, `successful_vs_genuine` and `previously_important_head`. |
| 15/16 | interpretable anomaly detector on the 36 stage-13 features (`exp16lib/anomaly.py`). The reference is clean qrel 2/3 documents of the training queries; per-head z-scores give an abnormal-head count with fixed `\|z\| > 2`; evaluation is 5-fold query CV with seed 42. The threshold T and the top-k heads are chosen on training folds only. Stage 16 attributes the excess abnormality of successful attacks to heads and ablates the detector to its top-k heads. No trained model. |

Every stage resumes safely through `status.json` (the Exp 01/14 convention)
and accepts `--force`.
- Model stages record the manifest sha256 and refuse to mix populations.
- Stage 02 resumes per query; stages 03, 07, 09, 11 and 13 resume per attack.
- Stage 03 accepts `--attack-start/--attack-end` for deterministic SLURM
  chunking.

```bash
bash bash/run_tests.sh                    # pytest (CPU)
bash bash/run_smoke_test.sh               # full pipeline on configs/smoke.yaml -> outputs_smoke/
bash bash/run_all.sh configs/default.yaml # full run (submit via SLURM, see run_all.sh header)
```

## Paired extension (stages 17–21)

**Question.** For the *same* judged document, when a successful attack raises
the monoT5 score, do its late-encoder query–document head similarities become
more abnormal than those of its own padded control? This removes the
document-population confound in stages 15–16, where attacked documents were
compared with *other*, genuinely relevant documents.

| Item | Definition |
|------|------------|
| Instance | (qid, docid, attack configuration): every judged attackable DL19 base pair × all 105 attacks, read directly from `ecir24-adversarial-evaluation/runs/injected/dl19/*.gz.tsv` with Exp 01's `load_attacked_tsv` (not `selected_examples.jsonl`) |
| Groups | qrel 2/3 (1,760 base pairs) and qrel 0 (2,262), always analysed separately; qrel 1 and unjudged dropped |
| Pair | padded control → attacked input (`encode_attack_and_control`; same length and positions) |
| Success | `delta_score = score_attack − score_control > 0` (logit true − logit false, fresh forward) |
| Features | 36 encoder heads L9–L11, pre-`o_proj`, cos(mean query-text, mean document) (stage 13 metric) |
| Reference (fold f) | controls of successful qrel 2/3 instances whose qid ∉ fold f, pooled over all 105 attacks; per-head μ, σ |
| Abnormal | \|z\| > 2; abnormal_count = # of 36 heads |
| CV | 5-fold by query, `anomaly.query_folds` over all judged attackable qids, seed 42 |
| Detector | positive = attacked input, negative = its own control; top-k heads, ranking, threshold T_k and k* chosen on inner folds of the training queries only |

```
17_prepare_paired_manifest -> 17_paired_manifest/{base_pairs.jsonl.gz, attacks/{attack}.jsonl.gz, alignment_failures.csv,
                              attack_population_summary.csv, fold_map.json, old_outputs_fingerprint.json, provenance.json}
18_run_paired_forward      -> 18_paired_forward/per_attack/{attack}/rows.csv.gz  (score + 25 checkpoints + 36 heads,
                              control and attack, one forward; per-attack resume; cross-check vs Exp 01 / stage 03 / stage 13)
19_paired_anomaly          -> 19_paired_anomaly/{paired_instances.csv.gz, reference_fits.csv, per_head_paired.csv,
                              paired_count_summary.csv, count_distribution.csv, document_level{.csv.gz,_summary.csv},
                              low_high_summary.csv, per_attack_paired.csv, success_counts.csv, summary.json}
20_paired_detector         -> 20_paired_detector/{topk_detector_metrics.csv, detector_fold_details.csv, selected_heads.csv,
                              detector_summary.json}
21_plot_paired             -> plots/paired/fig_paired_*.png, 21_paired_report/{plot_data/*.csv, final_summary.json,
                              old_outputs_check.json}
```

Run with `bash bash/run_paired.sh configs/default.yaml` (SLURM command in the
script header). Smoke with `bash bash/run_paired_smoke.sh` (4 attacks, 15
queries, ≤ 4 pairs per group per query → `outputs_paired_smoke/`). Stages
17–21 write only new directories; stage 21 re-hashes every pre-existing Exp 16
output and fails if any changed. Design choices: DECISIONS 45–52.

**All 12 layers (144 heads).** `bash bash/run_paired.sh configs/paired_all_layers.yaml`
(smoke: `bash bash/run_paired_all_layers_smoke.sh`). Same method, reuses the
stage-17 manifest, writes `18_paired_forward_all_layers/` …
`21_paired_report_all_layers/` and `plots/paired_all_layers/` (DECISIONS 53).

**Token-level exploration sample (stage 22).** `scripts/22_extract_token_sample.py`
stores, for ~250 judged base pairs (≤ 3 qrel 2/3 + ≤ 3 qrel 0 per query, seed 42) ×
(clean + a balanced 12-attack subset, control and attack), the per-token pre-`o_proj`
output of all 144 encoder heads (fp16, `outputs/22_token_sample/`, ~12 GB). Try new
metrics there with `exp16lib.token_sample.TokenSample`; promising ones are computed on
the fly over the full population later (pooled vectors are never bulk-stored).

```python
from exp16lib.token_sample import TokenSample
ts = TokenSample("outputs/22_token_sample")
for s in ts.iter(kind="control", relevance_group="relevant"):
    h = s["heads"][:, 11, 2].astype("float32")          # [tokens, 64], layer 11 head 2
    q, d = h[s["query_mask"]], h[s["doc_mask"]]         # query-text / document tokens (control: no insertion slots)
```

**Activation-statistics screen (stage 24).** `scripts/24_activation_stats.py`
(`exp16lib/activation_stats.py`; CPU, ~6 min, no forward pass) computes per head
and token region (query, full document, attacked original-document tokens,
inserted tokens) the mean token L2 norm, mean, per-token variance, max |x|,
effective dimensionality, top-token energy share and normalised token-energy
entropy on the stage-22 sample. It compares clean relevant vs non-relevant
documents, successful attacks vs their own padded control, and inserted vs
original (and size-matched original) tokens, and runs an exploratory PCA of the
L9–L11 pooled document head representation (2304-d, not the residual stream).
Outputs: `outputs/24_activation_stats/` (DECISIONS 54).

## Reused code (imported, never copied)

| Exp | Module | Used for |
|-----|--------|----------|
| 01 | `src.model_utils` | loading, prompt format, padded-control construction |
| 01 | `src.attack_registry` | attack registry |
| 01 | `src.scoring` / `score_from_encoding` | smoke score check |
| 03 | `headlib.run_utils` | config with `attacks.inherit_from` |
| 06 | `exp6lib.spans.find_query_and_doc_spans` | query/document spans |
| 13 | `exp13lib.head_lists` | canonical important encoder/decoder heads (stages 09/10) |
| 14 | `exp14lib.run_utils` | status.json helpers |

New functionality lives in `exp16lib/`.

See `DECISIONS.md` for the locked design choices (1–37) and the implementation
choices (38+).
