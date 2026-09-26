# Experiment 11 — Per-Head Activation Patching + Ablation (monoT5 Encoder)

The direct encoder-side analog of Experiment 3 (decoder head patching, 288 head-slots).
Explicitly deferred as out-of-scope in Experiment 3's and Experiment 6's Limitations
sections; this experiment closes that gap: **12 encoder layers × 12 heads = 144
head-slots**, whole-sequence patching (not per-position — that is Experiment 6's job).

Builds on:
- Experiment 1 (`../01_monot5_layer_patching/`): model loading, scoring, the 105-attack
  grid, padded-control (Type B) construction, alignment, and the fwd/rev/combined effect
  formulas — all **imported**, not copied.
- Experiment 3 (`../03_monot5_head_patching_ablation/`): the per-head patching mechanism
  (intervene on the `.o` projection's input, where head *h* is the contiguous
  `[h·d_kv:(h+1)·d_kv]` slice) and the zero-vs-mean/control ablation distinction — mean
  ablation is primary throughout, since Experiment 3 established zero ablation
  over-attributes importance. The batched-over-heads trick is re-derived for the encoder
  (see `exp11lib/engine.py`'s module docstring): Experiment 3 also gets encoder-output
  reuse (decoder-only patches), which does **not** apply here — the patch happens inside
  the encoder's own forward computation, so it must re-execute per (layer, direction).
- Experiment 6 (`../06_monot5_query_doc_patching/`): query/document span-finding
  (`exp6lib.spans`, via `exp6lib.run_utils.build_example_inputs`) and attack-token
  position exposure (`ExampleInputs.attack_span_indices`) are reused directly, per this
  experiment's prompt. Also the source of the layer-9 document→query attention-mass
  anomaly (1.42× uniform, head-averaged) this experiment cross-references at per-head
  granularity (Step 4), and the template-position sub-additivity finding (0.44×) whose
  pattern Step 3 predicts should repeat here.

## Steps

| Step | Script | Question |
|---|---|---|
| 1 | `scripts/01_encoder_head_patch_canonical.py` | Which of the 144 head-slots are causally important, canonical attack (n=100)? |
| 2 | `scripts/02_encoder_head_patch_full_sweep.py` | Does that hold across the full 105-attack grid (n=10/attack)? |
| 3 | (same script, after the sweep) | Do flagged heads' individual effects sum additively to the whole-layer effect? |
| 4 | `scripts/03_layer9_head_attention_crossref.py` | Do the heads with high document→query attention mass overlap with the causally important ones? |
| 5 | `scripts/04_attack_token_hub_analysis.py` | Does the attack token actively attend out to the query (hub), or is contamination passive? |
| — | `scripts/05_aggregate_and_plot.py` | Heatmap, top-10 table, zero-vs-mean scatter, box plot, combined figure, written summary. |

## Reuse policy

Each experiment folder in this project is self-contained; cross-experiment imports only
happen where a prompt explicitly calls for reuse (see `scripts/03`/`04`'s `EXP6_DIR`
`sys.path` insertion, matching Experiment 10's established pattern). No file outside this
directory was modified.

## Running

```bash
conda activate advseq2seq
bash bash/run_encoder_head_patching.sh          # full pipeline (resume-safe)
pytest tests/                                    # correctness checks (tiny random T5, CPU)
```

See `DECISIONS.md` for design choices not obvious from the code, and
`encoder_head_patching_experiment_report.tex` for the full writeup.
