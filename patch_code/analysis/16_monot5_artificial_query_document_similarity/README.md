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
```

Every stage resumes safely through `status.json` (the Exp 01/14 convention)
and accepts `--force`.
- Model stages record the manifest sha256 and refuse to mix populations.
- Stage 02 resumes per query and stage 03 per attack.
- Stage 03 accepts `--attack-start/--attack-end` for deterministic SLURM
  chunking.

```bash
bash bash/run_tests.sh                    # pytest (CPU)
bash bash/run_smoke_test.sh               # full pipeline on configs/smoke.yaml -> outputs_smoke/
bash bash/run_all.sh configs/default.yaml # full run (submit via SLURM, see run_all.sh header)
```

## Reused code (imported, never copied)

| Exp | Module | Used for |
|-----|--------|----------|
| 01 | `src.model_utils` | loading, prompt format, padded-control construction |
| 01 | `src.attack_registry` | attack registry |
| 01 | `src.scoring` / `score_from_encoding` | smoke score check |
| 03 | `headlib.run_utils` | config with `attacks.inherit_from` |
| 06 | `exp6lib.spans.find_query_and_doc_spans` | query/document spans |
| 14 | `exp14lib.run_utils` | status.json helpers |

New functionality lives in `exp16lib/`.

See `DECISIONS.md` for the locked design choices (1–37) and the implementation
choices (38+).
