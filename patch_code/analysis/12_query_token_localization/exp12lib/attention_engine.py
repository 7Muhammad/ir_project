"""
exp12lib/attention_engine.py
==============================
Part A (descriptive): attack-token <-> query-word attention, both
directions, control vs. attack, at two resolutions:
  1. layer-level (averaged over all 12 encoder heads)
  2. the previously identified important encoder heads (Experiment 11,
     Section 4.5 -- loaded read-only via exp12lib.important_heads; NEVER
     re-selected here from attention magnitude).

Reused, unchanged (not duplicated)
------------------------------------
  exp6lib.attention.compute_encoder_attentions -- one encoder forward pass
      with output_attentions=True, returns one (1, n_heads, L, L) tensor per
      layer. Cheap: attention analysis needs exactly 2 such forward passes
      per example (control, attack) -- no patched re-runs.

What is new here
------------------
Experiment 6's Part 1 (exp6lib/attention.py) measures normalized
query<->document MASS (a fraction of a uniform baseline) on the CLEAN vs.
ATTACKED inputs, contiguous spans only. This experiment needs a different
quantity per the prompt: RAW attention mass (not baseline-normalized)
between the (possibly non-contiguous) attack-token index set and each
individual query-word span, on the padded-CONTROL vs. ATTACKED inputs
(so "attack token" positions exist and are comparable in both: real attack
tokens in the attack run, masked pad tokens at the identical indices in the
control run), split into a "total mass across all N injected attack tokens"
and a "mean mass per injected attack token" so raw repetition count (1-5)
does not by itself make a repeated attack look like it interacts more
strongly with the query.

Row convention: attn[i, j] = how much position i (as attention *source*/
query row) attends to position j (as *key*). "attack tokens -> query word"
sums, for each attack-token row i, attention into the word's key columns;
"query word -> attack tokens" sums, for each query-word-subtoken row i,
attention into the attack span's key columns, then averages over the word's
own subtokens (never summed -- a word's raw span length is a tokenization
artifact, not a signal, matching how the causal engine treats a word as one
unit regardless of subtoken count).
"""

from __future__ import annotations

from typing import Dict, List

import torch

from exp6lib.attention import compute_encoder_attentions
from exp12lib.important_heads import ImportantHead
from exp12lib.query_words import QueryWordSpan


def _row_block_sums(attn: torch.Tensor, indices_a: List[int], indices_b: List[int]) -> torch.Tensor:
    """Per-row (over `indices_a`) sum of attention mass into `indices_b`. Shape (len(indices_a),)."""
    if not indices_a or not indices_b:
        return torch.zeros(len(indices_a))
    block = attn[indices_a][:, indices_b]
    return block.sum(dim=1)


def attack_query_word_stats(attn: torch.Tensor, word_indices: List[int], attack_indices: List[int]) -> Dict[str, float]:
    """
    Raw attack<->query-word attention mass for ONE (layer, head-or-avg)
    attention matrix `attn` (seq_len, seq_len).

    Returns dict with:
      attack_to_query_total     -- sum over all N attack-token rows of their
                                    attention mass into the word span
      attack_to_query_mean_per_token -- attack_to_query_total / N
      query_to_attack_total     -- word-subtoken-averaged attention mass
                                    into the WHOLE attack span
      query_to_attack_mean_per_token -- query_to_attack_total / N
    """
    n_attack = len(attack_indices)
    a2q_rows = _row_block_sums(attn, attack_indices, word_indices)
    a2q_total = a2q_rows.sum().item()
    a2q_mean = (a2q_total / n_attack) if n_attack else 0.0

    q2a_rows = _row_block_sums(attn, word_indices, attack_indices)
    q2a_total = q2a_rows.mean().item() if len(word_indices) else 0.0
    q2a_mean = (q2a_total / n_attack) if n_attack else 0.0

    return {
        "attack_to_query_total": a2q_total,
        "attack_to_query_mean_per_token": a2q_mean,
        "query_to_attack_total": q2a_total,
        "query_to_attack_mean_per_token": q2a_mean,
    }


def layer_averaged_matrix(attentions, layer_idx: int) -> torch.Tensor:
    """(seq_len, seq_len), averaged over all encoder heads at `layer_idx`."""
    return attentions[layer_idx][0].mean(dim=0)


def single_head_matrix(attentions, layer_idx: int, head_idx: int) -> torch.Tensor:
    """(seq_len, seq_len) for one specific (layer, head)."""
    return attentions[layer_idx][0, head_idx]


def run_attention_example(
    model,
    control_enc: Dict,
    attack_enc: Dict,
    words: List[QueryWordSpan],
    attack_span_indices: List[int],
    layers: List[int],
    important_heads: List[ImportantHead],
    device,
) -> List[Dict]:
    """
    Compute Part A's per-(layer, head-granularity, query-word) attention rows
    for one example. Two forward passes total (control, attack) --
    everything else is tensor slicing, no re-running the model.

    Returns
    -------
    List[Dict], one row per (layer, granularity, query_word):
        granularity == "layer_avg"    -> head field is None (averaged over all heads)
        granularity == "important_head" -> head field is the head index
    """
    ctrl_attn = compute_encoder_attentions(model, control_enc, device)
    atk_attn = compute_encoder_attentions(model, attack_enc, device)
    n_attack = len(attack_span_indices)

    important_by_layer: Dict[int, List[int]] = {}
    for h in important_heads:
        important_by_layer.setdefault(h.layer, []).append(h.head)

    rows: List[Dict] = []
    for layer_idx in layers:
        ctrl_avg = layer_averaged_matrix(ctrl_attn, layer_idx)
        atk_avg = layer_averaged_matrix(atk_attn, layer_idx)

        for w in words:
            ctrl_stats = attack_query_word_stats(ctrl_avg, w.token_indices, attack_span_indices)
            atk_stats = attack_query_word_stats(atk_avg, w.token_indices, attack_span_indices)
            rows.append(_make_row(layer_idx, "layer_avg", None, w, n_attack, ctrl_stats, atk_stats))

        for head_idx in important_by_layer.get(layer_idx, []):
            ctrl_h = single_head_matrix(ctrl_attn, layer_idx, head_idx)
            atk_h = single_head_matrix(atk_attn, layer_idx, head_idx)
            for w in words:
                ctrl_stats = attack_query_word_stats(ctrl_h, w.token_indices, attack_span_indices)
                atk_stats = attack_query_word_stats(atk_h, w.token_indices, attack_span_indices)
                rows.append(_make_row(layer_idx, "important_head", head_idx, w, n_attack, ctrl_stats, atk_stats))

    return rows


def _make_row(layer_idx, granularity, head_idx, w: QueryWordSpan, n_attack, ctrl_stats, atk_stats) -> Dict:
    row = {
        "layer": layer_idx,
        "granularity": granularity,
        "head": head_idx,
        "query_word_index": w.index,
        "query_word_text": w.text,
        "content_or_stopword": w.content_or_stopword,
        "matched_or_unmatched": w.matched_or_unmatched,
        "word_group": w.word_group,
        "num_attack_tokens": n_attack,
    }
    row["attack_to_query_control"] = ctrl_stats["attack_to_query_total"]
    row["attack_to_query_control_mean_per_token"] = ctrl_stats["attack_to_query_mean_per_token"]
    row["attack_to_query_attack"] = atk_stats["attack_to_query_total"]
    row["attack_to_query_attack_mean_per_token"] = atk_stats["attack_to_query_mean_per_token"]
    row["attack_to_query_delta"] = atk_stats["attack_to_query_total"] - ctrl_stats["attack_to_query_total"]

    row["query_to_attack_control"] = ctrl_stats["query_to_attack_total"]
    row["query_to_attack_control_mean_per_token"] = ctrl_stats["query_to_attack_mean_per_token"]
    row["query_to_attack_attack"] = atk_stats["query_to_attack_total"]
    row["query_to_attack_attack_mean_per_token"] = atk_stats["query_to_attack_mean_per_token"]
    row["query_to_attack_delta"] = atk_stats["query_to_attack_total"] - ctrl_stats["query_to_attack_total"]
    return row
