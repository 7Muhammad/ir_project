# Experiment 2 — Decisions Log

## Scope

- **Part 2 intervention scope restricted to Experiment-3-flagged heads only**
  (not all 288 decoder head-slots, and no encoder heads — see below).

- **In-sample attack set only.** The mean-diff direction is computed and
  tested on the same pool of examples per tier (grid_a: top-N-per-attack
  across all 105 attacks; grid_b: top-N of `relevant_start_5`). This
  establishes feasibility, not generalization to held-out attacks. **Do not
  present these results as a generalizing defense claim.**

- **Both locations and both granularities computed for Part 1**, per the
  original design: encoder self-attention and decoder cross-attention,
  whole-vector (per layer) and per-head. See the extension below for a third
  location added to Part 1.

## Deviations from the literal task spec (and why)

1. **Folder path**: `patch_code/analysis/02_monot5_mean_diff_intervention/`
   instead of the spec's literal `experiments/0X_mean_diff/`. This repo has
   no top-level `experiments/` directory — every numbered experiment lives
   under `patch_code/analysis/0N_monot5_<name>/` (01, 03, 06, 07 exist). "02"
   was the free, literal match for "Experiment 2."

2. **Flagged-head list**: no pre-existing "flagged heads" artifact exists in
   Experiment 3 (its README defers this explicitly). The spec's estimate of
   "~33 head-slots" at `combined_effect_mean > 0.02` does not match any
   version of the data on disk: grid_a (pooled over all 105 attacks) gives
   **22 heads**, grid_b (canonical attack only) gives 11, their union is 26.
   **User decision (2026-07-12): use grid_a, threshold 0.02 → 22 heads**
   (21 `decoder_cross_attn` + 1 `decoder_self_attn`, layer 11 head 3).

3. **`decoder_self_attn` added to Part 1's scope.** The spec names two
   locations ("encoder self-attention output, decoder cross-attention
   output"). The confirmed 22-head flagged list contains one
   `decoder_self_attn` head (L11 H3). Omitting `decoder_self_attn` from
   Part 1 would silently drop that head from Part 2's intervention scope.
   Since Experiment 3's `headlib/head_hooks.py` already treats
   `decoder_self_attn` and `decoder_cross_attn` identically (same
   `.o`-projection mechanism), extending Part 1 to three components costs
   nothing extra and preserves full coverage of the flagged set.

4. **Activations are recomputed, not read from a persisted cache.** Neither
   Experiment 1 nor Experiment 3 persist raw activations to disk: Exp1's
   multi-attack pipeline deletes `outputs/attacks/{attack}/activations/`
   after patching (`cleanup_activations_after_patching: true`), and Exp3
   never wrote per-head activations to disk at all (kept in-memory per
   example). Part 1 therefore recomputes clean/control/attack activations
   itself via new hook code (`exp2lib/direction_hooks.py`) that imports
   Exp1's/Exp3's hook *mechanisms*, not their cached *data*. Zero changes to
   `01_monot5_layer_patching/` or `03_monot5_head_patching_ablation/`.

5. **Direction-fitting design — two-tier, mirrors Experiment 3 (user
   decision, 2026-07-12).** One global direction per (component, layer[,
   head]), pooled from the top-`n_examples_per_attack` examples per attack
   across all 105 attacks — used for the Grid A breadth intervention, tested
   on that same pool. A second, attack-specific direction pooled from the
   top-N examples of `relevant_start_5` only — used for the Grid B depth
   intervention on that same attack.

6. **Fitting pool == test pool, in both tiers.** The direction for a tier is
   fit on exactly the same examples used to test the intervention for that
   tier (Experiment 1's `selected_examples.jsonl`, already sorted by
   `attack_delta_vs_control` desc, top-N taken). This is the strictest,
   simplest form of "in-sample" and avoids inventing a separate
   fitting-sample-size choice not specified anywhere in the task.

7. **Encoder self-attention pooling rule** (undocumented in the original
   task — encoder activations carry a real sequence dimension, unlike
   decoder heads which are always a single vector per example since monoT5
   scoring runs one decoder step). Both whole-vector and per-head encoder
   activations are **mean-pooled over valid (`attention_mask == 1`)
   positions** before being folded into the running mean, giving one
   direction vector per example per (layer[, head]).

8. **Direction-norm-vs-Exp3-ranking correlation is decoder-heads-only.**
   Experiment 3 never scored encoder heads (explicitly "out of scope" in its
   config), so there is no causal ranking to correlate `encoder_self_attn`
   directions against. This is a structural fact, not a bug.

## Priority

Per the task: the defense test (2a, subtract) is prioritized over the
sufficiency test (2b, add), since the project's goal is attack prevention.
Both are implemented and run together by default (`bash/run_all.sh`), but
`scripts/02_run_interventions.py --intervention defense` can run 2a alone.
