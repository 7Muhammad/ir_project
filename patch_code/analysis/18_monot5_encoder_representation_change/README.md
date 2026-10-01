# Experiment 18 — Encoder representation change (padded control → attacked input)

**Question:** where across the 12 monoT5 encoder layers does the representation of
an attacked input diverge from its padded control? The analysis compares the
full residual stream (768-d), separately for:

- the query;
- the original document;
- the whole sequence.

This experiment is representational, not causal. It shows where divergence
appears and what changes. It says nothing about which layer *causes* the score
change; the activation-patching experiments (Exp 01/03/11/13) supply that
evidence.

Model `castorini/monot5-base-msmarco` (fp32, no TF32), TREC DL19.

## Population (read-only from Exp 16)

- **Instances:** every aligned instance of Exp 16 stage 17
  (`16_…/outputs/17_paired_manifest`).
  - Judged base pairs: 1,760 qrel 2/3 ("relevant") and 2,262 qrel 0
    ("non-relevant"); qrel 1 and unjudged are excluded.
  - 43 queries × 105 attacks (7 tokens × 3 positions × 5 repetitions).
  - 422,206 aligned instances. The 104 Exp 01 alignment failures (15 random
    attacks) are logged in Exp 16 and excluded.
- **Scores and success:** reused from Exp 16 stage 18 (`18_paired_forward`).
  - `delta_score = score_attack − score_control`, and
    `successful ⇔ delta_score > 0` (the project definition).
  - 131,314 successful instances (55,006 relevant, 76,308 non-relevant).
  - Stage 01 recomputes the scores in its own forward only as a check.

## States

| state | hook | HF `hidden_states` |
|---|---|---|
| `embedding` (reference) | pre-hook on `encoder.block[0]` | `[0]` |
| `L00`…`L10` | output of `encoder.block[L]` | `[L+1]` |
| `L11` | output of `encoder.block[11]`, **before** `final_layer_norm` | not returned by HF |
| `final_norm` (reference) | `encoder.final_layer_norm` output | `[12]` = `last_hidden_state` |

Main figures use L00–L11 only. The embedding state is never mixed with L0.

## Regions

Notation: *A* = inserted positions; *P* = non-inserted positions. Removing *A*
from the attacked prompt gives the clean prompt exactly.

| region | control | attack | token-wise |
|---|---|---|---|
| `query` | query text | same positions | yes |
| `orig_doc` | clean passage tokens (via *P*) | same positions, injected excluded | yes |
| `whole_shared` | *P*: query + passage + template (`Query:`, `Document:`, `Relevant:`, `</s>`) | same positions | yes |
| `whole_full_visible` | every visible token (= *P*) | every token, **including injected** | no |

`whole_full_visible` mixes propagated change with the composition and length
effect of the added tokens. Always read it alongside `whole_shared`.

## Metrics

- **Mean-pooled state:**
  - `1 − cos`, computed as ½‖x/|x| − y/|y|‖² to avoid cancellation;
  - normalized L2 = ‖y − x‖ / max(‖x‖, 1e-6).
- **Token-wise:** the same two metrics on aligned positions; per instance the
  mean, median, and q10 of cos (q90 of L2).
- **Linear CKA, raw and query-centred:** population level, computed exactly from
  streamed float64 Gram statistics.
  - It is never correlated with per-instance Δscore.
  - For query-centring, each condition is centred on its OWN per-query mean.
- **Groups:** success (successful / unsuccessful / all) × relevance (all /
  relevant / non-relevant). The main view is successful × all.
- **Uncertainty:** 95% CI from a query-level bootstrap (2,000 replicates).
- **Correlation with Δscore:** Spearman ρ, pooled; within attack
  configuration; and per query.

## Storage

The user's rule (2026-09-30) is to persist only what is < 2 GB.

| option | size | kept |
|---|---|---|
| per-instance scalar metrics (23.6M rows, zstd Parquet) | ≈0.8 GB | **yes** |
| mean-pooled 768-d vectors (fp16) | ≈54 GB | no |
| token-level hidden states (fp16) | ≈1.6 TB | no |

## Run

```
bash bash/run_tests.sh      # 18 unit tests (CPU)
bash bash/run_smoke.sh      # 4 attacks x 24 instances, CPU, ~1.5 min -> outputs_smoke/
# full run (L40, ~35 min forward + CPU analysis), from /home/ghoummaid/IR:
bash run_job.sh --job-name monot5_exp18_repr --gpu-type L40 --gpu-count 1 --cores 8 --time 08:00:00 \
  --output-dir ./patch_code/analysis/18_monot5_encoder_representation_change/slurm_logs \
  --command "cd /home/ghoummaid/IR/patch_code/analysis/18_monot5_encoder_representation_change && bash bash/run_all.sh"
```

## Outputs

- `outputs/01_repr_change/`
  - `per_attack/*.parquet`: one row per instance × state × region, with qid,
    docid, attack, qrel, success, scores, and all metrics.
  - `cka.csv`, `population.json`, `per_attack_population.csv`, `checks.json`.
- `outputs/02_analysis/`
  - `layer_summary.csv`: mean, CI, median, n_instances, n_queries per
    state × region × metric × group.
  - `correlations.csv`, `cka.csv`, `population.json`.
  - `plots/fig1_cosine_by_layer.png`, `fig2_normalized_l2_by_layer.png`,
    `fig3_linear_cka_by_layer.png` (main);
    `fig4_successful_vs_unsuccessful.png`, `fig5_relevant_vs_nonrelevant.png`,
    `fig6_change_vs_delta_score.png` (secondary).

## Code

- `exp18lib/repr_change.py`: regions, integrity checks, metrics, hooks, CKA
  (direct + streaming), bootstrap.
- `exp18lib/config.py`: the Exp 17 config convention; the model block is read
  from the Exp 16 config.
- Imported from Exp 16 (never copied):
  - `exp16lib.inputs.encode_attack_and_control` (the Exp 01 padded-control
    construction);
  - `exp16lib.paired` (manifest constants);
  - `exp16lib.run_utils` (model loading, status).

See `DECISIONS.md`.
