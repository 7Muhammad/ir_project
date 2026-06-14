# monoT5 Layer-Level Activation Patching

**Research question:**
> At the layer/component level, where does monoT5 carry the attack signal
> from the repeated `"relevant:"` prefix to the increased `"true"` logit?

This experiment uses **activation patching** (causal tracing) to identify
which encoder/decoder layers and component types are responsible for
propagating the adversarial score increase caused by keyword-stuffing attacks
on monoT5.

---

## Background

The original ECIR-24 paper
([Parry et al., 2024](https://github.com/Parry-Parry/ecir24-adversarial-evaluation))
showed that prepending repeated tokens like `"relevant: "` to a document
dramatically increases its monoT5 relevance score — even for documents that
are not relevant.  This experiment asks **why** that happens at the
mechanistic level.

---

## How monoT5 scoring works

monoT5 is a T5 model fine-tuned to classify relevance.  It takes a prompt:

```
Query: {query} Document: {passage} Relevant:
```

and predicts whether the next token is `"true"` (relevant) or `"false"` (not relevant).

At inference we do **not** run generation.  Instead:
1. Encode the prompt with the T5 encoder.
2. Create `decoder_input_ids = [[<pad>]]` (the decoder start token, id=0).
3. Run one decoder forward step.
4. Extract `logit("true")` and `logit("false")` from decoder position 0.
5. Compute `score = logit("true") − logit("false")`.

**Why logit("true") − logit("false")?**  
Taking the difference cancels shared additive bias and gives a calibrated
score: positive → model leans "relevant", negative → model leans "not relevant".
This is the standard monoT5 inference procedure from the original paper and
the Pygaggle library.

---

## The three input types

| Type | Description | Prompt |
|------|-------------|--------|
| **A. Original clean** | Unmodified document, no prefix | `Query: Q Document: P Relevant:` |
| **B. Padded control** | Pad tokens with `attention_mask=0` in prefix positions | same seq length as C, passage at identical positions |
| **C. Attacked** | Adversarial prefix | `Query: Q Document: relevant:×5 P Relevant:` |

**Why do we need the control (B)?**

Activation patching substitutes an entire hidden-state tensor
`(batch, seq_len, d_model)` from one forward pass into another.  If the
original clean document (A) and the attacked document (C) have different
token counts, their activation tensors have different `seq_len` and cannot
be swapped.

We solve this by using a **padded control**: the positions occupied by the
attack prefix in (C) are filled with `tokenizer.pad_token_id` and their
`attention_mask` bits are set to `0`.  This means:

- Both (B) and (C) have the **same sequence length** and activation shapes.
- Passage tokens appear at **identical absolute positions** in both sequences.
- The pad slots contribute **nothing** to the model's computation — no
  gradient, no attention weight, no key/value content.
- No additional meaningful word (like `"information:"`) is introduced that
  could independently shift monoT5's score.

The construction is verified by checking that the passage token IDs (after
the common prefix) are identical between the original and attacked tokenisations.  
A `ValueError` is raised if the SentencePiece boundary assumption breaks.

The original clean input (A) is still scored for reference, but is not used
in patching.  All patching comparisons are between **padded control (B)** and
**attack (C)**.

---

## Experiment stages

```
find_data  →  00_prepare_pairs  →  01_score_and_select
           →  02_cache_activations  →  03_run_layer_patching  →  04_plot_results
```

| Stage | Script | Output |
|-------|--------|--------|
| find_data | `bash/find_data.sh` | `outputs/data_discovery/data_candidates.txt` |
| 00 | `scripts/00_prepare_pairs.py` | `outputs/pairs/pairs.jsonl` |
| 01 | `scripts/01_score_and_select.py` | `outputs/scores/all_scores.csv`, `selected_examples.jsonl` |
| 02 | `scripts/02_cache_activations.py` | `outputs/activations/*.pt` |
| 03 | `scripts/03_run_layer_patching.py` | `outputs/patching/layer_patching_results*.csv` |
| 04 | `scripts/04_plot_results.py` | `outputs/plots/*.png` |

---

## Quick start

```bash
cd patch_code/analysis/monot5_layer_patching

# Install dependencies
pip install -r requirements.txt

# Step 1: Discover available data files
bash bash/find_data.sh configs/default.yaml

# Step 2: Edit configs/default.yaml if needed (see Troubleshooting below)
# Easiest: set upstream_repo_path to the cloned ECIR-24 repo

# Step 3: Run the full experiment
bash bash/run_all.sh configs/default.yaml
```

Or run individual stages:
```bash
bash bash/run_prepare_pairs.sh     configs/default.yaml
bash bash/run_score_and_select.sh  configs/default.yaml
bash bash/run_cache_activations.sh configs/default.yaml
bash bash/run_layer_patching.sh    configs/default.yaml
bash bash/run_plots.sh             configs/default.yaml
```

---

## Output files

```
outputs/
  data_discovery/
    data_candidates.txt         -- list of all candidate data files found
  pairs/
    pairs.jsonl                 -- normalised (qid, docid, query, passage) pairs
  scores/
    all_scores.csv              -- original, control, and attack scores for all pairs
    selected_examples.jsonl     -- pairs where attack_score > control_score
  activations/
    {qid}_{docid}_control.pt    -- cached control activations (all layers)
    {qid}_{docid}_attack.pt     -- cached attack activations (all layers)
  patching/
    layer_patching_results_detailed.csv  -- per-example results
    layer_patching_results.csv           -- aggregated (mean/std per layer/component)
  plots/
    attack_delta_hist.png
    forward_layer_component_heatmap.png
    reverse_layer_component_heatmap.png
    combined_layer_component_heatmap.png
```

---

## How to interpret the heatmaps

Each heatmap has:
- **Y axis**: layer index (0 = earliest, 11 = last for monoT5-base)
- **X axis**: component type
  - `Enc Self-Attn` — encoder self-attention
  - `Enc MLP` — encoder feed-forward
  - `Dec Self-Attn` — decoder self-attention
  - `Dec Cross-Attn` — decoder cross-attention (attends to encoder output)
  - `Dec MLP` — decoder feed-forward
- **Colour**: patching effect (normalised)
  - ≈ 0 → this component carries little of the attack signal
  - ≈ 1 → this component fully explains the score change
  - > 1 → over-recovery or amplification

**Forward heatmap** (`forward_layer_component_heatmap.png`)  
> We run the *control* input but inject the *attack* activation at one (layer, component).  
> High value → this component is **sufficient** to raise the score (injecting attack here moves score from control to attack level).

**Reverse heatmap** (`reverse_layer_component_heatmap.png`)  
> We run the *attack* input but inject the *control* activation at one (layer, component).  
> High value → this component is **necessary** for the score increase (removing attack here cancels the effect).

**Combined heatmap** (`combined_layer_component_heatmap.png`)  
> `min(forward, reverse)`.  A bright cell is both sufficient *and* necessary — the attack signal flows through that component.

---

## Troubleshooting

**No data files found:**  
Set `upstream_repo_path` in `configs/default.yaml` to the absolute path of
the cloned ECIR-24 repo.  The script will find `data/bm25_19.jsonl`
automatically.

**Multiple candidate files found:**  
The discovery report lists all candidates.  Set the appropriate path
explicitly in `configs/default.yaml` (e.g. `bm25_run_path`).

**Format of `prepared_pairs_path`:**  
JSONL (one dict per line) or CSV/TSV with columns:
`qid`, `docid`, `query`, `passage` (or `text` instead of `passage`).

**Format of `bm25_run_path`:**  
Standard TREC run format: `qid Q0 docid rank score tag` (space-separated, 6 columns).
Can be `.gz` compressed.

**Passage text not found:**  
Set `passages_path` to the MS MARCO passage collection file (TSV: `docid\ttext`,
or JSONL with `id`/`contents` fields).

**Prefix token-length mismatch error:**  
If you change `attack_prefix` in the config, you must also change
`control_prefix` so that both tokenise to the same number of tokens.
Check with:
```python
from transformers import T5Tokenizer
tok = T5Tokenizer.from_pretrained("castorini/monot5-base-msmarco")
print(tok.encode("relevant: relevant: relevant: relevant: relevant: ", add_special_tokens=False))
print(tok.encode("information: information: information: information: information: ", add_special_tokens=False))
```

**Re-running one stage:**  
Each script is independent.  Just run the desired stage's bash script with
the config path.  Outputs are overwritten if they already exist.

---

## Not implemented (future work)

- Head-level activation patching (which attention heads carry the signal?)
- Token-level patching (which token positions carry the signal?)
- Path patching (causal circuit identification)
- Multiple attack types
- LLM rewriting attacks
- monoT5 training / fine-tuning
- BM25 retrieval from scratch

---

## Testing and validation

Before running the full experiment, verify that the code works correctly
using the three-level test strategy below.

### Level 1 — Unit tests

Unit tests check **individual pieces** of the code in isolation: tokenization,
scoring, padded-control construction, hook mechanics, and the effect formulas.
Most tests run in seconds; the tests that load the model take ~30 s on first run
(cached afterwards).

```bash
conda activate advseq2seq
bash bash/run_tests.sh
```

The test files and what they check:

| File | What it checks |
|------|----------------|
| `tests/test_patching_math.py` | Forward/reverse effect formulas with fake scores. No model needed — very fast. |
| `tests/test_tokenization.py` | "true" and "false" are single tokens; passage alignment in control/attack. |
| `tests/test_padded_control.py` | attention_mask=0 for pad slots; sequence lengths equal; passage tokens identical. |
| `tests/test_scoring.py` | `score_from_encoding` returns a finite float; no gradient accumulation. |
| `tests/test_hooks.py` | Cache hook stores the right shape; patch hook does not crash; hook cleanup. |

To run only the fast math tests (no model download needed):
```bash
pytest tests/test_patching_math.py -v
```

### Level 2 — Smoke test

The smoke test runs the **full pipeline end-to-end** on 5 pairs and 2 selected
examples using `configs/smoke.yaml`.  It checks that every stage completes and
that all expected output files are created.

This is the fastest way to confirm the pipeline works before committing GPU time.

```bash
conda activate advseq2seq
bash bash/run_smoke_test.sh configs/smoke.yaml
```

Outputs go to `outputs_smoke/`.  The script automatically verifies that these
files exist:
- `outputs_smoke/pairs/pairs.jsonl`
- `outputs_smoke/scores/all_scores.csv`
- `outputs_smoke/scores/selected_examples.jsonl`
- `outputs_smoke/patching/layer_patching_results.csv`
- `outputs_smoke/plots/*.png` (4 files)

### Level 3 — Sanity check (real-data diagnostics)

After running the smoke test or the full experiment, use the sanity-check script
to print statistics and spot suspicious patterns:

```bash
# Inspect smoke-test outputs
bash bash/run_sanity_check.sh configs/smoke.yaml

# Inspect full experiment outputs
bash bash/run_sanity_check.sh configs/default.yaml
```

The script prints:
- Number of pairs, scored examples, and selected examples
- Mean original / control / attack scores and their differences
- Percentage of examples where attack outscores control
- Mean forward and reverse patching effects by component type
- Warnings if anything looks wrong (empty selection, all-NaN effects, etc.)

### Recommended workflow

```bash
# 1. Run unit tests (fast, catches code-level bugs)
bash bash/run_tests.sh

# 2. Run tiny end-to-end smoke test (catches pipeline-level bugs)
bash bash/run_smoke_test.sh configs/smoke.yaml

# 3. Inspect smoke outputs (are scores and effects plausible?)
bash bash/run_sanity_check.sh configs/smoke.yaml

# 4. Run the real experiment (full data, full patching sweep)
bash bash/run_all.sh configs/default.yaml

# 5. Inspect real outputs
bash bash/run_sanity_check.sh configs/default.yaml
```

