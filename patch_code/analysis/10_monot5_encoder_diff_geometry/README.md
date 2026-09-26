# Experiment 10 — Encoder-Output Difference Geometry

Precursor/companion analysis to the head-localization (Experiment 3) and
mean-diff-steering (Experiment 2) experiments. Characterizes the geometry
of the encoder hidden-state difference between attacked and
padded-control inputs at layers 9/10/11, to answer: **is a single global
steering direction even the right kind of tool**, before more compute is
spent assuming it is?

## Research questions

1. Is the encoder-output attack-control difference low-rank at layer 11
   (and how does that compare to layers 9-10)?
2. Is the dominant difference direction consistent across different attack
   tokens/positions/repetitions, or is it attack-specific?
3. Is the difference concentrated at the injected-token positions
   themselves, or does it spread into other-document and query positions
   by layer 11?

## Reuse

Nothing here re-derives model loading, scoring, padded-control
construction, or token alignment — all imported directly from Experiment
1's `src` package. Query/document span detection is reused from Experiment
6's `exp6lib.spans`. See `DECISIONS.md` for exactly what's new versus
reused (in particular: the exact injected-attack-token sub-span, which
Experiment 6 computes internally but never exposes).

## Usage

```bash
# 1. Smoke test (2 attacks, n=5) -> outputs_pilot/
bash bash/run_pilot.sh

# 2. Primary run (15 strongest attacks by mean_attack_delta, n=100) -> outputs/
bash bash/run_all.sh

# 3. Optional extension (all 105 curated attacks, n=100) -> outputs_full_grid/
bash bash/run_full_grid.sh

# Unit tests (tiny T5Config + real tokenizer, no GPU)
bash bash/run_tests.sh
```

Each `run_*.sh` chains the five pipeline stages:

```
scripts/01_extract_diffs.py          -> outputs/diffs/{attack}/diffs.npz
scripts/02_pca_summary.py            -> outputs/geometry/{attack}/pca_summary.csv, top_directions.npz
scripts/03_cross_attack_cosine.py    -> outputs/geometry/cross_attack_cosine_matrix_layer{9,10,11}.csv
scripts/04_position_concentration.py -> outputs/geometry/position_concentration.csv
scripts/05_make_plots.py             -> outputs/plots/*.png
```

All stages are resume-safe (attack-level `status.json` + output-file
existence check); re-run with `--force` to redo from scratch.

## Outputs

- `outputs/geometry/{attack}/pca_summary.csv` — one row per (attack,
  layer): `variance_explained_top{1,3,10}`, `effective_rank`, `n_positions`.
- `outputs/geometry/cross_attack_cosine_matrix_layer{L}.csv` — attacks ×
  attacks cosine similarity of sign-aligned top-1 directions, one file per
  layer.
- `outputs/geometry/position_concentration.csv` — one row per (attack,
  layer, position_tag): `mean_diff_norm`, `std_diff_norm`, `n_positions`.
- `outputs/plots/scree_plot.png`, `cross_attack_cosine_heatmap.png`,
  `position_concentration_bars.png`, `effective_rank_vs_layer.png`.

## Report

`encoder_diff_geometry_experiment_report.tex` — full write-up matching the
style of Experiments 1-9.
