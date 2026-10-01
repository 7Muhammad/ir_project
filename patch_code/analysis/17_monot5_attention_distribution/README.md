# Experiment 17 — Encoder attention-distribution statistics

Exploratory screen: do genuine relevance or successful attacks change **how** monoT5 encoder
attention is distributed (entropy, concentration, effective support, locality, attention
involving the injected tokens)? Earlier work (Exp 16 report) looked at raw query↔document attention
**mass** and found that it did not identify the causally important heads. This experiment looks
at the **shape** of the attention distribution instead.

It is observational only: no detector, no patching. The causal evidence is still Exp 11/13.

## Input

The input is exactly the Exp 16 stage-22 token sample (`../16_…/outputs/22_token_sample/`,
read-only):

- 253 base pairs from 43 queries: 126 qrel 2/3 and 127 qrel 0 (qrel 1 excluded).
- 12 attacks, giving 6,325 sequences: 253 clean, 3,036 padded controls and 3,036 attacked.
- 1,026 successful attacks (`delta_score > 0`).

The cached `input_ids` and masks are re-run through the encoder with `output_attentions=True`, in
fp32 with eager attention. Every attention matrix is reduced to scalars on the fly and then
discarded; no attention matrices are ever written to disk.

## Definitions (`exp17lib/attention_stats.py`)

**Key scopes**
- `full_visible`: every visible key, including template tokens and, in attacked sequences, the
  injected tokens.
- `shared_key`: keys visible in both the control and the attack, i.e. the attacked sequence with
  its injected tokens removed as keys.

In both scopes each row is renormalised to sum to 1. For clean and control sequences the two
scopes are identical; stage 01 checks this.

**Source regions**
- `query`
- `doc`
- `orig` (document without the inserted tokens)
- `ins` (inserted tokens; attacked sequences only)

Template tokens are never a source region.

**Metrics** (computed per source row, then averaged over the region)
- `entropy_norm` (primary) and raw `entropy`
- `max_attn`, which is the same as top-1 mass
- `top3`, `top5`
- `neff`, `neff_norm`
- `dist`, `dist_norm`
- `local_mass` (|i−j| ≤ 5)

**Injected-token mass** (`full_visible` scope)
- Directions: query↔ins and orig↔ins, plus query↔orig as a reference within the same sequence.
- Each direction is reported raw and token-count-normalised:
  `mass / (|target| / N_visible)`, where 1 means uniform attention over tokens.

## Run

```bash
bash bash/run_tests.sh                    # unit tests (synthetic attention)
bash bash/run_smoke.sh                    # 6 base pairs -> outputs_smoke/ (CPU ok)
cd /home/ghoummaid/IR && bash run_job.sh --job-name exp17_attn --gpu-type L40 --gpu-count 1 --cores 8 \
  --time 24:00:00 --output-dir ./patch_code/analysis/17_monot5_attention_distribution/slurm_logs \
  --command "cd /home/ghoummaid/IR/patch_code/analysis/17_monot5_attention_distribution && bash bash/run_all.sh configs/default.yaml"
```

## Stages and outputs (`outputs/`)

| stage | script | outputs |
|---|---|---|
| 01 | `scripts/01_attention_stats.py` | `attention_statistics.npz` (`R [seq, region, scope, metric, 144]`, `M [seq, direction, raw/norm, 144]`), `attention_statistics.csv.gz`, `injected_mass.csv.gz`, `sequences_meta.csv`, `validation.json`, `mask_inspection.txt` |
| 02 | `scripts/02_analyze.py` | `relevance_attention_summary.csv`, `paired_attack_attention_summary.csv`, `inserted_attention_summary.csv`, `inserted_source_shape_summary.csv`, `inserted_by_factor.csv`, `causal_alignment.csv`, `summary_table.md`, `injected_table.md`, `plots/` |

In the plots, canonical encoder heads (the 18 Exp 13 senders, loaded through
`exp16lib.heads.encoder_heads`) are drawn with a box in the heatmaps and with bold tick labels on
the 144-head axes.
