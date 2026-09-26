# Score-vs-Rank Experiment — Decisions Log

## Folder numbering

`09_monot5_score_vs_rank/`, not `08`. `08` is deliberately left open:
`monot5_decoderlens_rank/` occupies that conceptual position (it calls
itself "this experiment" in its own report, positioned after Experiments
1/3/6/7, but never assigns itself a number on disk) and the task's
instructions are explicit not to rename or otherwise touch other
experiments' folders.

## DecoderLens: code reused, precomputed run outputs NOT reused

`monot5_decoderlens_rank/`'s own outputs (`outputs/attacks/{attack}/...`)
were not used as a data source, for two concrete reasons found during
research:

1. **Coverage gap.** DecoderLens's 226 successfully-processed attacks are
   missing 29 of the curated 105 — specifically every `relevant_*` and
   `true_*` attack, including `relevant_start_5`, the single canonical
   attack used as "Grid B" throughout Experiments 2 and 3. Using its
   outputs directly would silently drop over a quarter of the curated
   grid, with the most important canonical attack among the missing ones.
2. **Target cap conflict.** DecoderLens caps targets at
   `max_target_examples_per_attack: 100`. This task explicitly requires
   "no filtering... up to 500 pairs/attack" to avoid survivorship bias.

Instead: **DecoderLens's code** is reused directly —
`src/data_loading.py` (`Candidate`, `load_attack_tsv`, `build_candidate_sets`)
and `src/ranking.py` (`compute_rank`) — both confirmed to have zero
internal dependencies beyond the stdlib. Target scores come from
Experiment 1's own `outputs/attacks/{attack}/scores/all_scores.csv`
(already computed, all 105 curated attacks present, 496-500 rows each) —
zero new forward passes needed for targets at all.

## Candidate scoring is one global pass, not per-attack

Verified empirically before implementing: all 105 curated attacks'
`all_scores.csv` files share **exactly one** distinct 42-query qid-set
(not 105 different samples), and BM25 rank (hence top-100 candidate
membership) is byte-identical for a given query across every attack's TSV
(the underlying BM25 run is replicated into every attack file, never
recomputed). So the ~4,200 candidate clean scores (42 queries x up to 100
candidates) are computed exactly once, against a single attack's TSV, and
reused for every attack's rank computation — not recomputed 105 times.

## The `src`/`src` package name collision

Experiment 1's importable package and DecoderLens's importable package are
both literally named `src` at the top level. Adding both directories to
`sys.path` is unsafe: `import src` resolves to whichever gets imported
first and is cached in `sys.modules`; later `sys.path` reordering cannot
undo that. `exp9lib/decoderlens_import.py` loads DecoderLens's
`data_loading.py` and `ranking.py` directly by file path via
`importlib.util.spec_from_file_location`, under private module names
(`_decoderlens_data_loading`, `_decoderlens_ranking`), which never touches
the `src` name at all. Experiment 1's `src.*` is imported normally via the
standard `sys.path.insert` pattern used throughout this project. Neither
experiment's package was renamed.

(Implementation note: the loader must register the module in
`sys.modules` under its private name *before* calling `exec_module` —
`dataclasses` resolves string type annotations via
`sys.modules[cls.__module__]`, so skipping this step breaks
`Candidate`'s dataclass fields the first time they're used.)

## Targets outside their query's original top-100

The majority of Experiment 1's sampled targets (a spot check on
`relevant_start_5` found 451/500) have an original BM25 rank greater than
100 — Experiment 1 samples from up to 1000 documents per query, much
deeper than DecoderLens's top-100 candidate cutoff. `compute_rank` is
agnostic to whether the target is an original top-100 member; it simply
counts how many of the (up to) 100 candidates outscore the target's
substituted score, giving a well-defined rank in `[1, 101]` regardless.
This is intentional, not a gap: the task explicitly asks to measure the
score/rank relationship "across both positive and negative effects
without survivorship bias," which requires evaluating every example, not
just ones that started inside the candidate set.

## No SLURM job required

Total new model inference for this entire experiment is ~4,200 forward
passes (the one-time candidate scoring in script 01); everything else
(scripts 02-04) is pure CSV/dict/plotting logic with zero forward passes.
This runs in minutes even on the CPU-only login node, so no SLURM
submission was used (unlike Experiments 2/3, whose intervention grids cost
~150k forward passes each).
