# monoT5 DecoderLens Rank Experiment

DecoderLens-style analysis of where Parry / ECIR-24 keyword-stuffing attacks
become **readable** inside monoT5's encoder.

This is a sibling of `../01_monot5_layer_patching/` but asks a different question:

| Experiment | Question |
|---|---|
| `01_monot5_layer_patching` (activation patching) | *Where* does the attack become **causally important** inside the model? |
| `monot5_decoderlens_rank` (this experiment)   | *When* (which encoder layer) does the attack become **readable** from the encoder representation? |

This is **not** activation patching. There is no head-level, component-level, or
causal patching here. We only re-read a single forward pass's intermediate
encoder representations.

## Main idea

Normal monoT5 scoring uses only the final encoder output:

```
final encoder hidden states -> decoder -> logit(true), logit(false)
score = logit(true) - logit(false)
```

DecoderLens forces the decoder to cross-attend to **each encoder layer** in turn:

```
encoder layer 0  hidden states -> decoder -> score_0
encoder layer 1  hidden states -> decoder -> score_1
...
encoder layer 12 hidden states -> decoder -> score_12
```

Each intermediate (raw block) hidden state is passed through the encoder's
`final_layer_norm` before the decoder (the top layer is already normalised by
the encoder, so it is used as-is — this makes layer 12 match the normal monoT5
score exactly; verified by the final-layer sanity check).

The layerwise scores are then used to **re-rank a fixed candidate set** and
track how the target passage's rank changes across encoder layers.

## Layer indexing (monoT5-base)

```
layer 0  = embedding / pre-encoder-block representation
layer 1  = after encoder block 1
...
layer 12 = after encoder block 12   (== normal monoT5 last hidden state)
```

13 layers total (`num_encoder_blocks + 1`).

## Input variants

For each target query-document pair:

```
original:        Query: q Document: original_passage  Relevant:
padded_control:  Query: q Document: [PAD slots] passage Relevant:   (mask=0 on PAD)
attack:          Query: q Document: attacked_passage  Relevant:
```

`padded_control` and `attack` have the **same length** with the real passage
tokens aligned; inserted spans are detected with `difflib.SequenceMatcher`
(prefix-walk + suffix-check shortcut) so it works for start / end / random
attacks, repetition counts 1–5, and single / multi / compound tokens. The main
comparison is **attack vs padded_control**.

## Ranking

For each (attack, query, target doc, variant, layer):

1. Replace the target passage with one variant (original / padded_control / attack).
2. Keep all other candidates clean.
3. Score every candidate with DecoderLens layer `l`.
4. Sort by score; record the target's rank (rank 1 = best).

The candidate set is the top-`candidate_top_k` (default 100) BM25 candidates
taken from the attack TSV's `rank` column — **BM25 is never recomputed**.

## Multi-attack by default

The experiment discovers and runs **all** manual injection attacks
(`attacks.mode: all`). For debugging, `include` / `include_pattern` still work.
Weak / negative attacks are **kept** (`selection.mode: all_aligned`) so their
layerwise rank curves can be compared too.

## Layout

```
configs/    default.yaml, smoke.yaml
src/        attack_registry, data_loading, padded_control, decoderlens,
            scoring, ranking, plotting, comparison, utils
scripts/    01_discover_attacks .. 07_run_all_decoderlens_attacks
bash/       run_all_decoderlens_attacks.sh, run_compare_decoderlens_attacks.sh
outputs/    generated artifacts
```

## Running

```bash
conda activate advseq2seq

# Smoke test (1 attack, 20 candidates, 5 targets) -> outputs_smoke/
bash bash/run_all_decoderlens_attacks.sh configs/smoke.yaml --force

# Full run over all discovered attacks -> outputs/
bash bash/run_all_decoderlens_attacks.sh configs/default.yaml
```

On the SLURM cluster (see repo memory / `run_job.sh`):

```bash
bash run_job.sh --job-name "monot5_decoderlens" --gpu-type L40 --gpu-count 1 \
  --cores 8 --time 12:00:00 --output-dir ./slurm_logs \
  --command "cd /home/ghoummaid/IR/patch_code/analysis/monot5_decoderlens_rank && \
             bash bash/run_all_decoderlens_attacks.sh configs/default.yaml"
```

## Outputs

Per attack (`outputs/attacks/{attack_name}/`):

```
pairs/target_pairs.jsonl
scores/layerwise_target_scores.csv
scores/layerwise_candidate_scores.parquet
ranks/layerwise_target_ranks.csv
ranks/layerwise_rank_summary.csv
plots/rank_over_layers.png
plots/rank_gain_over_layers.png
plots/score_delta_over_layers.png
plots/success_rate_over_layers.png
sanity/final_layer_equivalence.json
status.json
```

Cross-attack (`outputs/attack_comparison/`):

```
decoderlens_summary.csv
rank_gain_heatmap.png
score_delta_heatmap.png
success_rate_heatmap.png
top_attacks_rank_gain.png
top_attacks_score_delta.png
layer_of_first_positive_effect.png
```

`rank_gain_vs_control = control_rank - attack_rank` (positive = attack improved
rank). The main plot is `rank_over_layers.png` (x = encoder layer, y = rank,
y-axis inverted so rank 1 is at the top).
