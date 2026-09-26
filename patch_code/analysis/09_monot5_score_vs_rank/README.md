# Score-vs-Rank Comparison Experiment

Validates this project's score-based results (Experiment 1's
`logit(true) - logit(false)` metric) against Parry et al.'s ECIR-24
rank-based evaluation protocol, on the same curated 105-attack grid
(7 tokens x 3 positions x 5 reps) used throughout this project.

**Purpose:** show the two metrics correlate — average logit-confidence
change vs. rank movement/success rate — without claiming they must agree
on every point. A big score jump on a document that's still 80th in its
query's ranking is a real, separate phenomenon from a small score jump
that pushes a document from rank 3 to rank 1. See `DECISIONS.md` for
several research findings that shaped the implementation (a DecoderLens
output coverage gap affecting the canonical `relevant_start_5` attack, and
a package-name collision between two sibling experiments).

## Metrics

Per (attack, qid, docid):
```
delta_score = score_attack - score_control
delta_rank  = rank_control - rank_attack     # positive = moved toward rank 1
success     = 1 if delta_rank > 0 else 0
```
`rank_control`/`rank_attack` are the target's rank within its query's fixed
BM25 top-100 candidate set (DecoderLens's `compute_rank`) — every other
candidate keeps its clean score; only the target is swapped for its
control/attack score.

Per attack config (105 rows): `mean_delta_score, mean_delta_rank,
success_rate, spearman_per_attack`.

**Headline number**: `spearman_global` — Spearman rho between `delta_score`
and `delta_rank` pooled across every example from every attack (per-config
n is small and noisy; the pooled value is what should be reported, e.g.
"score change and rank change correlate at rho=X across N examples").

## Usage

```bash
conda activate advseq2seq
bash bash/run_smoke.sh              # 3 attacks, top-20 candidates, a couple minutes on CPU
bash bash/run_all.sh                # full 105-attack grid, configs/default.yaml
```

Or step by step:
```bash
python scripts/01_build_candidate_scores.py --config configs/default.yaml  # one-time, ~4,200 forward passes
python scripts/02_compute_score_rank.py     --config configs/default.yaml  # 0 forward passes
python scripts/03_aggregate.py               --config configs/default.yaml
python scripts/04_make_plot.py                --config configs/default.yaml
```

`pytest tests/` runs the unit tests (no model needed except for the
smoke/full run itself — `exp9lib` is otherwise pure CSV/dict logic,
tested with a monkeypatched scorer and a synthetic attack TSV).

## Outputs

```
outputs/
  candidates/candidate_scores.json          # {qid: {docid: clean_score}}, script 01
  per_example/{attack_name}.csv              # script 02, per (qid,docid) row
  aggregate/
    per_attack_summary.csv                    # 105 rows, script 03
    global_spearman.json                       # the headline number
  plots/score_vs_rank_scatter.png             # script 04, the one deliverable plot
```

## Deliberately not implemented

- No held-out attack split — this is a full-grid descriptive/correlational
  analysis, not a train/test design.
- No per-layer rank tracking (that's DecoderLens's own research question,
  Experiment 8/`monot5_decoderlens_rank/`) — this experiment only uses
  ordinary final-layer monoT5 scoring.
