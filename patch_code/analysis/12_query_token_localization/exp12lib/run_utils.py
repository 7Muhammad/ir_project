"""
exp12lib/run_utils.py
=======================
Config loading, example-pool selection (with SEPARATE, independently
configurable attention/causal sample sizes), and per-example input assembly
for Experiment 12.

Reused, unchanged (not duplicated)
------------------------------------
  exp6lib.run_utils.build_example_inputs  -- Type A/B/C encodings +
      query_span / doc_span_original / doc_span_control_attack /
      attack_span_indices, alignment-checked.
  exp6lib.run_utils.score_and_select_examples -- fallback path when no
      Experiment-1 reuse file exists for an attack.
  exp6lib.spans, exp6lib.template_positions -- via structural_masks.py.
  src.attack_registry.AttackSpec / discover_attacks
  src.patching.SKIP_EPSILON (== 1e-4, the required success filter)

What is new here
------------------
Experiment 6/11's example-pool loader (`load_reused_examples`) simply slices
the top-N of an already min_delta=0.0-filtered, descending-sorted reuse
file. This experiment's prompt requires the stricter `delta > 1e-4` filter
(re-applied explicitly here, never assumed from the upstream file) AND two
INDEPENDENT sample-size caps (attention: up to 100/attack, causal: up to
10/attack) drawn from the SAME sorted pool -- so the causal set is always a
prefix of the attention set for a given attack, by construction, which keeps
the two analyses on directly comparable examples. Also attaches query-word
spans/classification (exp12lib.query_words) and structural-group masks
(exp12lib.structural_masks) to Experiment 6's ExampleInputs, since both
downstream engines (causal, attention) need them.
"""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import yaml
from transformers import T5Tokenizer

from src.attack_registry import AttackSpec
from src.patching import SKIP_EPSILON

from exp6lib.run_utils import build_example_inputs as _build_example_inputs_exp6
from exp6lib.run_utils import score_and_select_examples
from exp6lib.template_positions import get_template_positions

from exp12lib.query_words import QueryWordSpan, classify_query_words, find_query_word_spans
from exp12lib.structural_masks import build_structural_units


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(config_path: pathlib.Path) -> dict:
    """Load config, resolving attacks.inherit_from (Experiment 1's 105-attack grid)."""
    config_path = pathlib.Path(config_path).resolve()
    with open(config_path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    attacks_cfg = cfg.get("attacks") or {}
    inherit = attacks_cfg.get("inherit_from")
    if inherit:
        inherit_path = (config_path.parent / inherit).resolve()
        with open(inherit_path, encoding="utf-8") as fh:
            base_attacks = yaml.safe_load(fh)["attacks"]
        overrides = {k: v for k, v in attacks_cfg.items() if k != "inherit_from" and v is not None}
        cfg["attacks"] = {**base_attacks, **overrides}

    cfg["_config_dir"] = str(config_path.parent)
    return cfg


def resolve_cfg_path(cfg: dict, path_str: str) -> pathlib.Path:
    p = pathlib.Path(path_str)
    return p if p.is_absolute() else (pathlib.Path(cfg["_config_dir"]) / p).resolve()


# ---------------------------------------------------------------------------
# Example pool: reuse-first (Experiment 1's selections, re-filtered at 1e-4),
# fallback to fresh scoring. One pool per attack, sliced independently for
# attention vs. causal sample sizes by the caller.
# ---------------------------------------------------------------------------

def _load_reuse_pool(attack_name: str, reuse_dir: pathlib.Path) -> Optional[List[Dict]]:
    path = reuse_dir / attack_name / "scores" / "selected_examples.jsonl"
    if not path.exists():
        return None
    examples: List[Dict] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def get_example_pool(
    spec: AttackSpec, cfg: dict, model, tokenizer, true_id, false_id, device, pool_size: int,
) -> List[Dict]:
    """
    Return up to `pool_size` examples for `spec`, filtered strictly on
    `attack_delta_vs_control > SKIP_EPSILON` (1e-4, per the experiment
    prompt's success criterion -- NOT the looser min_attack_delta some
    upstream selection files used), sorted descending by delta.
    """
    reuse_dir_str = cfg["selection"].get("reuse_dir")
    pool: Optional[List[Dict]] = None
    if reuse_dir_str:
        pool = _load_reuse_pool(spec.attack_name, resolve_cfg_path(cfg, reuse_dir_str))

    if pool is None:
        n_rescore = max(pool_size, cfg["selection"].get("rescoring_pool_size", pool_size))
        pool = score_and_select_examples(
            spec, n_rescore, cfg, model, tokenizer, true_id, false_id, device,
        )

    filtered = [p for p in pool if p["attack_delta_vs_control"] > SKIP_EPSILON]
    filtered.sort(key=lambda x: x["attack_delta_vs_control"], reverse=True)
    return filtered[:pool_size]


# ---------------------------------------------------------------------------
# Per-example input assembly: exp6lib's ExampleInputs + word spans/masks
# ---------------------------------------------------------------------------

@dataclass
class Exp12ExampleInputs:
    status: str
    reason: str = ""
    control_enc: Optional[Dict[str, torch.Tensor]] = None
    attack_enc: Optional[Dict[str, torch.Tensor]] = None
    query_span: Tuple[int, int] = (0, 0)
    doc_span_control_attack: Tuple[int, int] = (0, 0)
    attack_span_indices: List[int] = field(default_factory=list)
    words: List[QueryWordSpan] = field(default_factory=list)
    structural_units: Dict[str, List[int]] = field(default_factory=dict)
    template_positions: Dict[str, List[int]] = field(default_factory=dict)


def build_exp12_example_inputs(
    tokenizer: T5Tokenizer, example: Dict, max_length: int, device: torch.device,
) -> Exp12ExampleInputs:
    base = _build_example_inputs_exp6(tokenizer, example, max_length, device)
    if base.status != "ok":
        return Exp12ExampleInputs(status="align_failed", reason=base.reason)

    word_result = find_query_word_spans(tokenizer, example["query"], base.query_span)
    if word_result.status != "ok":
        return Exp12ExampleInputs(status="align_failed", reason=f"query word spans: {word_result.reason}")
    words = word_result.words
    classify_query_words(words, example["passage"])

    attack_ids = base.attack_enc["input_ids"][0].tolist()
    template_positions = get_template_positions(
        tokenizer, attack_ids, base.query_span, base.doc_span_control_attack,
    )
    structural_units = build_structural_units(
        template_positions, base.query_span, base.doc_span_control_attack, base.attack_span_indices,
    )

    return Exp12ExampleInputs(
        status="ok",
        control_enc=base.control_enc,
        attack_enc=base.attack_enc,
        query_span=base.query_span,
        doc_span_control_attack=base.doc_span_control_attack,
        attack_span_indices=base.attack_span_indices,
        words=words,
        structural_units=structural_units,
        template_positions=template_positions,
    )
