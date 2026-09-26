# Experiment 7 — Logit Lens: Design Decisions

Merges what would have been separate Experiments 4 and 5 (query/doc/attack
composition of top-k, and attack-token tracking) with a general logit-lens
pass (conceptually "Experiment 1" of the logit-lens family — not to be
confused with `01_monot5_layer_patching`), since all three need the exact
same underlying infrastructure: capture a cross-attention contribution,
project it through the unembedding, read off top-k.

## Scope: cross-attention only

Not decoder self-attention, not the full residual stream. Chosen to
directly extend Experiments 1 and 3's finding that the attack signal is
concentrated in decoder cross-attention (Experiment 1: layer-level,
late-layer cross-attention; Experiment 3: 12/13 of the important head-slots
above the 0.02 combined-effect threshold are cross-attention heads).
**Revisit only if cross-attention's logit-lens results are ambiguous or
inconclusive** — e.g. if the attack token never appears in top-k anywhere
in cross-attention, self-attention or the full residual stream would be the
next place to look.

## What "post" means, precisely

The task specifies projecting the cross-attention sublayer's output
"post... i.e. what gets written into the residual stream" — this is the
sublayer's own **contribution**, `out_hidden - in_hidden`, computed from a
forward hook on the whole `T5LayerCrossAttention` module
(`model.decoder.block[i].layer[1]`), exactly the same quantity Experiment
1's `src/activation_hooks.make_cache_hook` already caches for patching. It
is NOT the accumulated hidden state after all prior layers, and NOT the
raw attention-weighted value before the output projection — it is
`dropout(EncDecAttention(...))` in eval mode (dropout is identity), i.e.
precisely what T5LayerCrossAttention.forward adds onto its input hidden
state.

## Unembedding: lm_head only, no final_layer_norm re-application

`castorini/monot5-base-msmarco` has `tie_word_embeddings=True` and
`scale_decoder_outputs=True`, so the model's own final readout is
`lm_head(sequence_output * d_model**-0.5)` (verified against the installed
transformers' `T5ForConditionalGeneration.forward` source). Experiment 7
reuses `model.lm_head` directly with the same `d_model**-0.5` scaling —
this is not a re-implementation, it is the model's own unembedding, applied
to a different input.

**`final_layer_norm` is deliberately NOT re-applied** to the isolated
contribution before this projection. `final_layer_norm` is calibrated for
the shape/scale of the *fully accumulated* residual stream; re-running it
on a single sublayer's raw contribution would rescale by that contribution's
own (much smaller, differently-distributed) RMS norm, inflating logit
magnitudes in a way that reflects the contribution's own variance rather
than anything about the model's actual computation. This is the standard
"direct logit attribution" convention (cf. TransformerLens's treatment of
per-component logit attribution) rather than classical full-residual logit
lens. If probing whole accumulated hidden states (not just cross-attention's
contribution) becomes useful later, that would be a different, explicitly
labeled analysis, not silently mixed into these numbers.

## Granularity: whole-block always, per-head only for Experiment-3-flagged heads

Whole-block logit lens (contribution summed over all 12 heads, i.e. the
module's actual output) runs for all 12 decoder layers, unconditionally.
Per-head logit lens is restricted to heads Experiment 3 flagged as
important — read live from
`../03_monot5_head_patching_ablation/outputs/grid_a/aggregated/head_summary.csv`
at run time (not hardcoded), filtered to `component == "decoder_cross_attn"`
(matching this experiment's cross-attention-only scope) and
`combined_effect_mean > flagged_head_threshold` (default `0.02`, the same
cutoff used descriptively in Experiment 3's report: 13/288 heads exceed it,
12 of those 13 already cross-attention). The threshold is a config value,
not hardcoded, so it can be revisited without a code change. A per-head
contribution is the head's slice of the `.o`-projection input (same
decomposition Experiment 3 uses), zero-padded to full `d_model` width
before projection through `lm_head` (the unembedding only accepts
`d_model`-wide vectors; a single head's raw `d_kv`-wide slice cannot be
projected on its own without first mapping it back through `W_o`, i.e. the
correct per-head contribution to the residual stream is
`W_o(zero_pad(head_h_slice))`, not the raw slice).

## Query/document/attack-token tagging

Each top-k token is tagged as exactly one of `attack`, `query`, `document`,
`stopword`, `other`, checked in that priority order. Attack-token check
comes FIRST, not last: the 7 attack-grid words ("relevant", "true", "false",
"information", "bar", "important", "relevance") are ordinary English words
that can legitimately co-occur in query or document text, and the entire
point of Analysis 4 is to isolate how much of top-k is specifically the
injected trigger word — if a coincidental query/document overlap silently
absorbed those matches into the `query`/`document` bucket instead, the
attack's apparent dominance would be systematically undercounted. Matching
is:

1. **Exact string match** (case-insensitive, SentencePiece `▁`-prefix
   stripped) against the token-level vocabulary of the query text, the
   document (clean passage) text, and the attack-token text respectively.
2. **Stem match fallback** for near-misses (plural/case/inflection
   variants) using a small dependency-free suffix-stripping stemmer
   (`exp7lib/tagging.py:simple_stem`) — **not** NLTK's Porter stemmer,
   because `nltk` is not installed in this project's environment
   (`advseq2seq`) and none of Experiments 1/3/6 required it; adding a new
   dependency for a fallback-only heuristic was judged not worth it. The
   stemmer is intentionally crude (strip trailing `s`/`es`/`ing`/`ed`/`ly`)
   — it only needs to catch the common inflection cases the task calls out
   (e.g. plural/case variants), not do linguistically correct stemming.
3. **Stopword** — a fixed, hardcoded ~180-word standard English stopword
   list (`exp7lib/tagging.py:STOPWORDS`), for the same dependency reason.
4. **Other** — everything else.

## Attack-token tracking: clean, control, AND attack runs

Logged for all three run types per the task's explicit requirement: clean
and control give the near-zero/noise-floor baseline; attack shows the
promotion. Without the baseline, a claim like "the attack token rises to
rank 3 by layer 9" has no reference point — rank 3 could be unremarkable if
random tokens land there on clean inputs too. `attack_token_rank` /
`attack_token_logit` are computed identically across all three run types
(same token id looked up in each run's own top-k / full logit vector), so
the three curves are directly comparable.
