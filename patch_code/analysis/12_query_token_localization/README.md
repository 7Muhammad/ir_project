# Experiment 12 — Query-Token Localization

**Question.** The positional-patching experiment (Experiment 6, `../06_monot5_query_doc_patching/`)
showed that a document-only attack propagates into the query representation:
query-only patching moves the score robustly at late encoder layers. But that
treats the query as one span. Here we ask: is that corruption spread evenly
across the query, or concentrated in particular words / word types?

    Part A (descriptive): attack-token <-> query-word attention, both
        directions, control vs. attack, at layer-averaged and
        previously-identified-important-head granularity.
    Part B (causal, MAIN RESULT): whole-encoder-self-attention-output
        patching at one query-word span at a time (all its SentencePiece
        subtokens patched jointly), forward/reverse/combined effect, exactly
        the metric definitions from Experiments 1/6/11.

No outcome (content > stopword, matched > unmatched, uniform, etc.) is
assumed — the implementation is built to distinguish between them, not to
confirm one.

## Reuse (imported, never copied)

| From | What |
|---|---|
| `../01_monot5_layer_patching/src/` | model loading, tokenizer, `score_from_encoding`, padded-control/attack construction + alignment (`src.alignment`), `SKIP_EPSILON` (== 1e-4, the success filter), activation-hook primitives, the 105-attack grid (`configs/multi_attack.yaml`, via `attacks.inherit_from`) |
| `../06_monot5_query_doc_patching/exp6lib/spans.py` | `query_span` / `doc_span` finding (alignment-checked prefix probing) |
| `../06_monot5_query_doc_patching/exp6lib/template_positions.py` | the 7 named template-token position groups (`Query`, `:`, `Document`, `:`, `Relevant`, `:`, `</s>`) |
| `../06_monot5_query_doc_patching/exp6lib/engine.py` | `make_positional_patch_hook` (position-masked encoder self-attention patch hook) |
| `../06_monot5_query_doc_patching/exp6lib/template_engine.py` | `build_template_position_mask` (arbitrary index-list → mask), reused directly for query-word spans and structural groups |
| `../06_monot5_query_doc_patching/exp6lib/attention.py` | `compute_encoder_attentions` (one encoder forward pass, all layers/heads) |
| `../06_monot5_query_doc_patching/exp6lib/run_utils.py` | `build_example_inputs` (Type A/B/C encodings), `score_and_select_examples` (rescoring fallback) |
| `../11_monot5_encoder_head_patching/outputs/top10_heads.csv` | the already-identified important encoder heads (Section 4.5) — **read-only**, never re-derived from attention here |

No file outside `12_query_token_localization/` was modified.

## Successful-instance filtering

An example is "successful" iff `attack_delta_vs_control > 1e-4` (== `src.patching.SKIP_EPSILON`,
the same skip guard every other patching experiment in this project uses).
This filter is applied **unconditionally** in `exp12lib.run_utils.get_example_pool`,
regardless of whether the example pool came from Experiment 1's reuse files
(`min_attack_delta: 0.0`, looser) or from fresh rescoring — never assumed
from an upstream selection file.

## Sample sizes (independently configurable)

`configs/default.yaml`:

```yaml
sampling:
  attention:
    n_examples_per_attack: 100   # up to 100/attack, cheap (2 fwd passes/example)
  causal:
    n_examples_per_attack: 10    # up to 10/attack for the initial breadth run
```

Both draw from the same delta-sorted per-attack pool (`get_example_pool`), so
the causal set is always a prefix of the attention set for a given attack.
Re-running causal patching at `n=30` or `n=100` requires editing
`sampling.causal.n_examples_per_attack` only — no code change (same code
path, different config; see `exp12lib/causal_engine.py`, which has no
knowledge of sample size at all).

## Word-span construction

A "query word" = one whitespace-delimited unit of the *original* query
string (`query.split()`). All of a word's SentencePiece subtokens are
patched jointly, as one intervention unit — never split, never patched one
subtoken at a time. Spans are located by the same alignment-checked
prefix-probing style `exp6lib.spans` already uses: each successive probe
`"Query: " + " ".join(words[:i+1])` must be an exact token-level prefix of
`"Query: " + query`, or the example is skipped as an alignment failure
(`exp12lib/query_words.py`). The union of all word spans is asserted to
exactly reproduce `query_span` (Sanity Check 1).

## Query-word classification

- **content vs. stopword** — a fixed, embedded copy of NLTK's
  `stopwords.words("english")` list (179 words; `exp12lib/stopwords.py`).
  Neither `nltk` nor `sklearn` is installed in the `advseq2seq` conda
  environment, so the list is hardcoded rather than adding a new runtime
  dependency (and a corpus-download step) for a single fixed word list.
- **matched vs. unmatched** — whether `normalize_word(query_word)` appears
  in `{normalize_word(t) for t in clean_passage.split()}`, where
  `normalize_word(w) = w.strip(string.punctuation).lower()`. Uses the
  **original clean document** (`example["passage"]`), never the attacked
  one. No semantic/synonym matching.

These combine into four word groups: `content_matched`, `content_unmatched`,
`stopword_matched`, `stopword_unmatched`.

## Structural position masks

Ten positions/groups derived entirely from `exp6lib.spans` +
`exp6lib.template_positions` (never re-probed or hardcoded):
`Query`, `Query:`, query content, `Document`, `Document:`, injected attack
tokens, original document tokens, `Relevant`, `Relevant:`, `</s>`. The 9
causal control groups (`exp12lib/structural_masks.py`) exclude `</s>`
(patching past EOS has no defined semantics here). `attack_tokens` /
`original_document` split `doc_span` using the exact inserted-token indices
already tracked by `build_padded_control_and_attack_encodings_general`
(`ExampleInputs.attack_span_indices`) — not re-derived.

## Example-balanced aggregation (main result)

A query with many words must not outweigh a query with few words merely by
word count. Procedure (`exp12lib/aggregation.py`):

1. average all words in a group **within one example** (one (attack, qid,
   docid, layer, group) value);
2. average those per-example values **across examples**.

Intervention-weighted (raw-row-average, word-count-biased) aggregates are
also saved as a secondary diagnostic. `tests/test_aggregation.py` has a
hand-computed worked example showing the two differ when word counts are
uneven.

## Checkpointing / resume

Per attack: `outputs/{causal,attention}/attacks/<attack_name>/{status.json,results.csv}`.
On resume (no `--force`), an example is skipped once its `(qid, docid)`
already has rows in `results.csv` — an attack interrupted mid-way only
re-does its last unwritten example, not previously-written ones (verified:
see the resume test described below). An attack whose `status.json` says
`"success"` is skipped entirely. Causal and attention are separate
pipelines/output trees, independently resumable, independently enable-able
(`enable.causal` / `enable.attention` in the config).

## Running

```bash
conda activate advseq2seq

# debug on one attack, tiny samples, outputs_smoke/:
bash bash/run_smoke_test.sh

# sanity checks 1-9 against real examples:
python scripts/00_sanity_checks.py --config configs/default.yaml --attack relevant_start_5 --n-examples 5

# full 105-attack grid:
bash bash/run_all.sh configs/default.yaml
# or, via SLURM:
bash /home/ghoummaid/IR/run_job.sh --job-name monot5_query_token_localization \
    --gpu-type L40 --gpu-count 1 --cores 8 --time 12:00:00 --output-dir ./slurm_logs \
    --command "cd /home/ghoummaid/IR/patch_code/analysis/12_query_token_localization && bash bash/run_all.sh configs/default.yaml"

pytest tests/   # correctness checks (real tokenizer, CPU, no GPU/model download needed)
```

## Sanity checks (script `00_sanity_checks.py`)

1. Union of word spans == query_span. 2. `Query`+colon disjoint from
   query-content mask. 3. `attack_tokens` mask == `ExampleInputs.attack_span_indices`.
4. `query_content` structural patch reproduces Experiment 6's `query_only`
   condition, computed via two independent code paths on the same example
   (`exp6lib.engine.run_encoder_self_attn_example` vs.
   `exp12lib.causal_engine.run_causal_patch_units_example`), asserted equal
   within `1e-5`. 5. Same equivalence check for a single template position
   (`Relevant`) against `exp6lib.template_engine`. 6. Multi-subtoken words
   patched as one contiguous span. 7. matched/unmatched reproduced by hand
   against the clean document. 8-9. read existing run outputs and confirm
   the causal breadth run never exceeds its configured `n`, and that
   attention/causal sample sizes are independent config keys.

All of 1-7 pass against real `relevant_start_5` examples (GPU, `castorini/monot5-base-msmarco`).

## Outputs

```
outputs/
  causal/attacks/<attack>/{status.json,results.csv,selected_examples.jsonl}
  attention/attacks/<attack>/{status.json,results.csv,selected_examples.jsonl}
  aggregates/*.csv        # scripts/03_aggregate.py
  plots/*.png             # scripts/04_make_plots.py — 5 main figures
  diagnostics/            # scripts/05_diagnostics.py — individual query-word
                           # inspection table + per-example word x layer pivots
```

## Connection to the next experiment (encoder-head @ query-word → decoder-head path patching)

Query-word spans/classification/masks are computed once
(`exp12lib.run_utils.build_exp12_example_inputs`) and are directly reusable:
`inputs.words` (spans + `content_or_stopword`/`matched_or_unmatched`/`word_group`),
`inputs.structural_units`, `inputs.attack_span_indices`. Combined with
Experiment 11's important encoder heads, a future path-patching experiment
can target `important encoder head @ important query-word position →
important decoder cross-attention head → score` without recomputing any of
this experiment's alignment/classification logic.
