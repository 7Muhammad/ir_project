# Experiment 14 — Phase 1 Held-Out Single-Head Mean-Diff Defense

**Research question:**
> Can a fixed mean-difference direction, learned from training examples at
> one causally important attention head, reduce keyword-stuffing attack
> effects on unseen query-document pairs and unseen attack configurations,
> without substantially changing monoT5's score on clean inputs?

This is **Phase 1**: one head at a time. Multi-head/joint interventions are
explicitly out of scope (deferred to Phase 2 — see DECISIONS.md item 11).
Evaluation uses monoT5's score, `logit("true") - logit("false")`, throughout
— no rank-based evaluation (Experiment 9 already validates score as a
practical proxy for rank movement).

## Motivation

Experiment 2 showed that subtracting an attack-minus-control mean-diff
direction at important decoder heads can move scores back toward control,
but:

- direction fitting and testing used the **same examples** (in-sample only)
- only the (now-superseded) earlier 22-head decoder list was intervened on
- **encoder heads were never tested**
- **clean-input damage was never evaluated**
- **unseen-attack generalization was never tested**

This experiment closes all five gaps: a proper train/validation/test split,
the current canonical head sets, encoder position-restricted steering,
mandatory clean-damage reporting, and attack-configuration-OOD evaluation.

## What is reused (imported, never copied)

| From | What |
|---|---|
| `01_monot5_layer_patching/src/model_utils.py` | model/tokenizer loading, `logit(true)-logit(false)` scoring, padded-control (Type B) construction (`build_padded_control_and_attack_encodings_general`), the monoT5 prompt template |
| `01_monot5_layer_patching/src/alignment.py` | token-level attack-insertion alignment |
| `01_monot5_layer_patching/src/attack_registry.py` | 105-attack grid discovery (7 tokens × 3 positions × 5 reps) |
| `01_monot5_layer_patching/src/patching.py` | `SKIP_EPSILON`, the `min(fwd_effect, rev_effect)` combined-effect formula |
| `01_monot5_layer_patching/outputs/pairs/pairs.jsonl` | the canonical 500-pair sample the train/val/test split is built on |
| `01_monot5_layer_patching/outputs/attacks/*/scores/selected_examples.jsonl` | per-attack successful-instance pool, with cached `control_score`/`attack_score`/`original_score` (no rescoring) |
| `02_monot5_mean_diff_intervention/exp2lib/direction_fit.py` | `DirectionAccumulator`, `pool_activation` (encoder mean-pool over valid positions; decoder single-step, no pooling) |
| `02_monot5_mean_diff_intervention/exp2lib/direction_hooks.py` | `cache_activations_for_direction` (one pass caches whole-vector + per-head activations for every layer/component), `get_encoder_o_proj` |
| `02_monot5_mean_diff_intervention/exp2lib/run_utils.py` | `save_directions`/`load_directions` (torch `.pt`), `save_norm_rows`, `build_clean_encoding` |
| `03_monot5_head_patching_ablation/headlib/head_hooks.py` | `head_geometry` (n_heads, d_kv), the `.o`-projection head-slice convention |
| `03_monot5_head_patching_ablation/headlib/engine.py` | `compute_encoder_states`/`decoder_pass_scores` (encoder-output reuse for decoder-only interventions) |
| `03_monot5_head_patching_ablation/headlib/run_utils.py` | `load_config` (`attacks.inherit_from` resolution), `resolve_cfg_path` |
| `06_monot5_query_doc_patching/exp6lib/spans.py` | `find_query_and_doc_spans` — query/document token-span boundaries |
| `06_monot5_query_doc_patching/exp6lib/template_positions.py` | `get_template_positions` — the 7 named template/prompt sub-regions |
| `13_monot5_encoder_decoder_path_patching/configs/heads/{encoder_senders,decoder_receivers}.json` | the canonical 18 encoder + 31 decoder candidate heads (see DECISIONS.md item 1) |

No previous experiment's code was modified. Sibling directories are added to
`sys.path` at the top of each script (the same convention Experiments 2/3/6
already use), never symlinked or copied.

## What is new

- **Pair-disjoint train/validation/test split** (`exp14lib/splits.py`),
  built *before* any attack-success filtering, seed 42, 60/20/20 by default,
  with every one of the 105 attack variants of a pair guaranteed to land in
  that pair's split.
- **Encoder per-head, position-restricted subtraction hook**
  (`exp14lib/hooks.py::make_encoder_masked_shift_pre_hook`) — combines
  Experiment 2's per-head slice math with Experiment 6's position-masked
  patch shape; batched over scales in one encoder forward pass.
- **Five encoder intervention-position conditions**
  (`exp14lib/position_masks.py`): document, query, template,
  query_document, all_valid — built from Experiment 6's own span/template
  logic, no new position definitions invented.
- **Clean-input (Type-A) defense-damage evaluation** — mandatory primary
  output, not optional.
- **Attack-configuration OOD evaluation**
  (`exp14lib/ood_folds.py`, `scripts/07_run_attack_ood.py`):
  leave-one-token-out (7 folds), leave-one-position-out (3), leave-one-
  repetition-out (5), each with its own fold-specific direction (fit on
  seen attacks only) and fold-specific scale selection (chosen on seen
  attacks' validation data only).
- **Normalized per-example recovery** and **denominator-pooled global
  recovery** metrics (`exp14lib/metrics.py`), never clipped.
- **Validation-only scale selection** with the smaller-scale tie-break rule,
  plus an explicit (not threshold-gated) recovery-vs-clean-damage curve
  (`exp14lib/scale_selection.py`, `scripts/05_select_scales.py`).

## Scope

- One head at a time (Phase 1). No multi-head/joint intervention.
- Score-based evaluation only (`logit(true) - logit(false)`). No rank
  evaluation.
- No model retraining, no learned/trained attack detector.
- No encoder-decoder path patching (that is Experiment 13's job).
- No position-specific direction *fitting* — only the intervention *site*
  is position-restricted (see DECISIONS.md item 5).

## Pipeline

```
scripts/
00_derive_candidate_heads.py   -> outputs/candidate_heads.json (18 encoder + 31 decoder)
01_prepare_splits.py           -> outputs/split_manifest.json
02_cache_baselines.py          -> outputs/baselines/  (per-split successful/clean example pools)
03_fit_directions.py           -> outputs/directions/iid/directions.pt   (TRAIN only)
04_run_validation.py           -> outputs/validation/validation_results.csv, validation_clean_damage.csv
05_select_scales.py            -> outputs/scale_selection/selected_scales.json
06_run_iid_test.py             -> outputs/test/iid_test_per_example.csv, clean_damage_per_example.csv
07_run_attack_ood.py           -> outputs/ood_test_per_example.csv, ood_clean_damage_per_example.csv
08_aggregate.py                -> outputs/full_recovery_summary.csv, full_clean_damage_summary.csv
09_make_plots.py                -> outputs/plots/figure_{a,b,c}_*.png
```

Every stage after 00/01 is resume-safe via a per-stage (or, for script 07,
per-fold) `status.json` — re-running `bash/run_all.sh` after a failure or
timeout skips completed stages unless `--force` is passed.

## Running

```bash
conda activate advseq2seq

# unit tests (no GPU, no full model download needed except tokenizer)
bash bash/run_tests.sh

# CPU-feasible smoke pipeline (tiny data, writes to outputs_smoke/)
bash bash/run_smoke_test.sh

# full run (GPU required) — see bash/run_all.sh header for the run_job.sh invocation
bash run_job.sh --job-name monot5_exp14_heldout_defense --gpu-type L40 --gpu-count 1 \
    --cores 8 --time 48:00:00 --output-dir ./slurm_logs \
    --command "cd /home/ghoummaid/IR/patch_code/analysis/14_monot5_heldout_mean_diff_defense && \
               bash bash/run_all.sh configs/default.yaml"
```

## Outputs (full run, `configs/default.yaml`)

See `outputs/` after a full run: `candidate_heads.json`, `split_manifest.json`,
`directions/{iid,leave_token_out,leave_position_out,leave_repetition_out}/`,
`validation/`, `scale_selection/`, `test/`, `ood_*`, `full_*_summary.csv`,
`plots/`. See DECISIONS.md for every deliberate choice not dictated by the
original task spec.
