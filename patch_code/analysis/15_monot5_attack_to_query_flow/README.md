# Experiment 15 — Attack-token → query information flow

**Question.** The 18 important encoder self-attention heads matter for the
keyword-stuffing attack on `castorini/monot5-base-msmarco`. Do they make it
work by *directly* moving information from the injected attack-token
positions into the query-token positions?

## Motivation: what earlier experiments did not show

- Exp 01/06: late encoder self-attention (layers 10–11) carries most of the
  attack effect. Patching the **query positions** of the residual stream
  recovers part of it, so the attack changes the query representations even
  though the injected tokens sit in the document.
- Exp 11: the effect is sparse across encoder heads, and 18 heads were
  flagged as important. Exp 12 localized the effect within query tokens.
  Exp 13 traced encoder → decoder paths.
- **Gap.** Query-position patching shows *that* the query states carry
  attack information, not *how* it got there. The route could be a direct
  attack→query attention message, a multi-hop path (attack → document →
  query), a template relay, or an effect in earlier layers. Whole-head
  patching (Exp 11) swaps a head's output at *every* position and for
  *every* source, so it cannot isolate one route either.
- **Attention weights alone are not enough.** A large `P[q, a]` says the
  head reads from the attack positions. It does not say the value read
  there differs from the control, nor that the resulting message changes the
  score. Exp 11 already found that most layer 10–11 heads with large causal
  effects have unremarkable attention mass. Exp 15 therefore intervenes on
  the message itself.

## The intervention: source-specific edge-message patching

For encoder layer L, head h, target position q, with post-softmax attention
`P[q, j]` and per-head values `V[j]` (d_kv = 64):

```
z(q) = Σ_j     P[q, j] V[j]      # the head's pre-o_proj output (slice h of the .o input)
m(q) = Σ_{a∈A} P[q, a] V[a]      # the part of z(q) that comes from the attack positions A
```

A contains **all** injected token positions of the example (every
repetition, every scattered span), taken from Exp 01's alignment. The
targets are the **query-text model-token positions**, taken from Exp 06's
spans; template tokens are excluded.

The **padded control** (Type B) is the matched donor. It has the same length
as the attacked input, but the attack positions hold pads with
attention_mask = 0. Its `m_control` is computed explicitly and is exactly 0,
because masked keys get probability 0.

| direction | receiver | edit at target q | question |
|---|---|---|---|
| forward | padded control | `z ← z_ctrl − m_ctrl + m_atk` | **Sufficiency:** does inserting only the attacked attack→query message recreate the attack? |
| reverse | attacked run | `z ← z_atk − m_atk + m_ctrl` | **Necessity:** does removing only that message remove the attack? |

The edit is applied to head h's slice of the `.o` input, i.e. before the
shared output projection. Nothing else changes: not other heads, not other
positions, and not the contributions of other sources, because there is no
attention renormalization. The model then runs forward normally, and the
score is `logit("true") − logit("false")` at the first decoder step.

**Conditions.** *All-query* (primary) edits every query position of the head
at once, as one explicit intervention. *Single-query-token* (descriptive)
edits one query position at a time, and is run for every position.

## Sample and scope

- **Attacks:** all 105 (7 tokens × start/end/random × reps 1–5), inherited
  from `01_monot5_layer_patching/configs/multi_attack.yaml`.
- **Examples:** successful instances only (`score_attack − score_control >
  SKIP_EPSILON`, imported from Exp 01), drawn from Exp 01's cached pools.
  Each attack's pool is shuffled with seed 42 and the first 50 are kept.
  Every attack has ≥ 80 successes, so the run has **5,250 examples**, mean
  8.0 query tokens. One immutable manifest is shared by every head,
  condition and direction.
- **Heads:** the 18 canonical encoder heads in
  `13_…/configs/heads/encoder_senders.json`. Each head is tested
  individually; there are no joint-head runs.
- **Baselines:** the cached Exp 01 scores. Fresh scores are recomputed as a
  consistency check only.

## Metrics, aggregation and statistics

```
δ = S_atk − S_ctrl
raw_fwd = S_ctrl,patched − S_ctrl        e_fwd = raw_fwd / δ
raw_rev = S_atk − S_atk,patched          e_rev = raw_rev / δ
e_combined = min(e_fwd, e_rev)           # headline; never clipped
```

- **Equal weight across attacks:** examples are averaged within each attack,
  then the 105 attack means are averaged.
- **95% percentile bootstrap CI** (10,000 replicates), resampling examples
  within each attack while keeping attacks fixed.
- **Primary tests:** 18 in total, one per head, on the all-query
  e_combined. Each is a one-sided sign-flip test on the 105 attack-level
  means (100,000 flips, `p = (k+1)/(N+1)`), followed by Benjamini–Hochberg
  at α = 0.05.
- **Whole-head reference (secondary):** Exp 11's whole-head patching,
  recomputed on the same manifest with the same baselines. The ratio
  `edge_combined / whole_head_combined` is reported only when
  `|denominator| > SKIP_EPSILON`, and is not tested.
- **Single-token results:** descriptive only. No tests, no word or category
  grouping, and the all-query effect is never assumed to equal the sum of
  single-token effects.

## Pipeline

| stage | script | output (under `outputs/`) |
|---|---|---|
| 00 | `00_prepare_samples.py` | `sample_manifest.jsonl`, `00_samples/{sample_summary.csv, compute_budget.json}` |
| 01 | `01_sanity_checks.py` | `01_sanity/sanity_report.json` (fails the pipeline on any failed check) |
| 02 | `02_run_all_query_edges.py` | `02_all_query/per_attack/{attack}/rows.csv.gz` |
| 03 | `03_run_single_query_edges.py` | `03_single_query/per_attack/{attack}/rows.csv.gz` |
| 04 | `04_run_whole_head_reference.py` | `04_whole_head/per_attack/{attack}/rows.csv.gz` |
| 05 | `05_aggregate.py` | `05_aggregate/` raw tables, per-attack × head and global summaries, edge vs whole-head, token summaries, baseline consistency |
| 06 | `06_statistics.py` | `06_statistics/head_statistics.csv` (CIs, p, BH q, reject) |
| 07 | `07_plot.py` | `plots/fig1…fig7` |
| 08 | `08_posthoc_robustness.py` | `08_posthoc/robustness_by_head.csv` — POST-HOC descriptive robustness (DECISIONS 42) |

`run_all.sh` runs the stages in the order 00, 01, 02, 04, 03, 05, 06, 07,
so the primary results finish before the largest (descriptive) stage.

**Resume and immutability.**
- Every stage writes a `status.json`, and completed units are skipped.
- The model stages resume per attack. A unit that was completed against a
  different manifest or head set is an error, not something to skip or mix.
- `--force` recomputes a stage.
- `--attack-start/--attack-end` split the model stages across SLURM jobs.

```bash
bash bash/run_tests.sh                               # pytest (CPU, ~seconds)
bash bash/run_smoke_test.sh                          # tiny full pipeline -> outputs_smoke/
bash bash/run_all.sh configs/default.yaml            # full run (use SLURM, below)

cd /home/ghoummaid/IR && bash run_job.sh --job-name monot5_exp15_query_flow --gpu-type L40 \
  --gpu-count 1 --cores 8 --time 24:00:00 \
  --output-dir ./patch_code/analysis/15_monot5_attack_to_query_flow/slurm_logs \
  --command "cd /home/ghoummaid/IR/patch_code/analysis/15_monot5_attack_to_query_flow && bash bash/run_all.sh configs/default.yaml"
```

## Code reuse

Earlier experiments are imported, never copied or modified (`exp15lib/__init__.py`
adds the sibling directories to `sys.path`):

- **Exp 01** (`src.*`): model loading, padded-control construction and
  alignment, `SKIP_EPSILON`, the attack registry.
- **Exp 03** (`headlib.run_utils.load_config`): `attacks.inherit_from`.
- **Exp 06** (`exp6lib.spans.find_query_and_doc_spans`): the query span.
- **Exp 11** (`exp11lib`): head geometry, `.o` hooks, decoder scoring, and
  whole-head patching.
- **Exp 13** (`exp13lib.head_lists.load_senders`): the 18 heads.
- **Exp 14** (`exp14lib.run_utils`): the status helpers.

New Exp 15 code lives in `exp15lib/`:
- `positions`: A and Q, with invariant checks.
- `edge_messages`, `edge_hooks`, `engine`: the batched intervention.
- `reference`: the unoptimized live-recompute reference.
- `whole_head`, `metrics`, `statistics`, `stage_runner`.

## Tests and sanity checks

`tests/` (35 tests) and stage 01 cover:
- **A** decomposition; **B** control message ≈ 0 and attack message ≠ 0;
  **C** a no-op donor leaves the score unchanged.
- **D/E/F** target, head and all-query isolation (bitwise).
- **G/H** the forward and reverse formulas.
- **I** attack alignment (start, end, random; reps 1 and > 1); **J** the
  query span; **K** sampling; **L** metrics.
- **M** edge patching with A = Q = all positions ≡ Exp 11 whole-head
  patching.
- **N** optimized batched engine ≡ live-recompute reference.
- Batch-size invariance, and fresh-vs-cached baselines.

## Deliberate non-goals

The following are left out on purpose; see DECISIONS.md, which lists every
locked choice.
- No 144-head scan, no unimportant, random or joint-head controls.
- No document-position, random-source or random-target controls.
- No attention renormalization.
- No Exp 12 token categories.
- No representation-space outcome metrics.
- No token-level significance testing.

## Interpretation limits

A positive edge result means direct attack→query communication **through
that head** mediates part of the attack. It does not show:
- that this is the only route;
- that the 18 heads jointly explain the attack;
- that attention magnitude explains importance;
- that token effects add linearly.

A small edge effect alongside a large whole-head effect is informative: the
head then acts through other positions (document or template), through
indirect routes, or downstream.
