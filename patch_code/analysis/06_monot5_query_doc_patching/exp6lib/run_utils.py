"""
exp6lib/run_utils.py
=====================
Config loading, example reuse, and per-example input assembly for
Experiment 6. Mirrors Experiment 3's reuse-first design (see
03_monot5_head_patching_ablation/headlib/run_utils.py) but is written fresh
here — Experiment 6 must be self-contained and Experiment 3's folder is not
to be modified or imported from.
"""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import yaml
from transformers import T5Tokenizer

from src.attack_registry import AttackSpec
from src.data_utils import load_attacked_tsv
from src.model_utils import build_monot5_input, build_padded_control_and_attack_encodings_general
from src.scoring import score_pairs_general, select_attacked_examples

from exp6lib.spans import QueryDocSpans, find_query_and_doc_spans


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(config_path: pathlib.Path) -> dict:
    """
    Load the experiment config, resolving `attacks.inherit_from` (points at
    Experiment 1's multi_attack.yaml 105-attack grid, same pattern as
    Experiment 3, so the grid definition is never copy-pasted).
    """
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
# Example reuse (same reuse-first policy as Experiment 3)
# ---------------------------------------------------------------------------

def load_reused_examples(attack_name: str, n_examples: int, reuse_dir: pathlib.Path) -> Optional[List[Dict]]:
    path = reuse_dir / attack_name / "scores" / "selected_examples.jsonl"
    if not path.exists():
        return None
    examples: List[Dict] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    examples.sort(key=lambda x: x["attack_delta_vs_control"], reverse=True)
    return examples[:n_examples]


def score_and_select_examples(spec: AttackSpec, n_examples: int, cfg: dict, model, tokenizer,
                               true_id: int, false_id: int, device) -> List[Dict]:
    data_cfg = cfg["data"]
    pairs = load_attacked_tsv(
        path=spec.path, max_pairs=data_cfg["max_pairs"],
        max_docs_per_query=data_cfg["max_docs_per_query"], seed=cfg["runtime"]["seed"],
    )
    scored, _ = score_pairs_general(
        pairs=pairs, model=model, tokenizer=tokenizer, true_id=true_id, false_id=false_id,
        max_length=cfg["model"]["max_length"], device=device, batch_size=cfg["runtime"]["batch_size_scoring"],
    )
    return select_attacked_examples(
        scored_pairs=scored, min_attack_delta=cfg["selection"]["min_attack_delta"],
        max_selected=n_examples, seed=cfg["runtime"]["seed"],
    )


def get_examples_for_attack(spec: AttackSpec, n_examples: int, cfg: dict, model, tokenizer,
                             true_id: int, false_id: int, device) -> List[Dict]:
    reuse_dir_str = cfg["selection"].get("reuse_dir")
    if reuse_dir_str:
        reused = load_reused_examples(spec.attack_name, n_examples, resolve_cfg_path(cfg, reuse_dir_str))
        if reused is not None:
            return reused
    return score_and_select_examples(spec, n_examples, cfg, model, tokenizer, true_id, false_id, device)


# ---------------------------------------------------------------------------
# Per-example input assembly: Type A/B/C encodings + spans in each indexing
# ---------------------------------------------------------------------------

@dataclass
class ExampleInputs:
    status: str                  # "ok" | "align_failed"
    reason: str = ""
    clean_enc: Optional[Dict[str, torch.Tensor]] = None       # Type A
    control_enc: Optional[Dict[str, torch.Tensor]] = None     # Type B
    attack_enc: Optional[Dict[str, torch.Tensor]] = None      # Type C
    query_span: Tuple[int, int] = (0, 0)                      # same for A/B/C
    doc_span_clean: Tuple[int, int] = (0, 0)                  # Type A indexing
    doc_span_control_attack: Tuple[int, int] = (0, 0)         # Type B/C indexing
    attack_span_indices: Optional[List[int]] = None           # exact inserted-token
                                                                # indices (Type B/C
                                                                # indexing), from
                                                                # AlignmentResult.inserted_positions
                                                                # -- may be non-contiguous
                                                                # (e.g. "random"-position attacks)


def build_example_inputs(
    tokenizer: T5Tokenizer,
    example: Dict,
    max_length: int,
    device: torch.device,
) -> ExampleInputs:
    """
    Build Type A (clean), Type B (padded control), Type C (attacked)
    encodings for one example, plus query/document spans in both the
    Type-A indexing and the shared Type-B/C indexing.

    `example` must have "query", "passage" (clean), "attacked_passage".
    """
    query, passage, attacked_passage = example["query"], example["passage"], example["attacked_passage"]

    control_enc, attack_enc, align_result = build_padded_control_and_attack_encodings_general(
        tokenizer=tokenizer, query=query, passage=passage, attacked_passage=attacked_passage,
        max_length=max_length, device=device,
    )
    if align_result.status != "ok":
        return ExampleInputs(status="align_failed", reason=f"attack alignment: {align_result.reason}")

    spans = find_query_and_doc_spans(tokenizer, query, passage, n_attack_tokens=align_result.n_inserted)
    if spans.status != "ok":
        return ExampleInputs(status="align_failed", reason=f"span alignment: {spans.reason}")

    clean_text = build_monot5_input(query, passage)
    clean_tok = tokenizer(clean_text, return_tensors="pt", max_length=max_length, truncation=True, padding=False)
    clean_enc = {"input_ids": clean_tok["input_ids"], "attention_mask": clean_tok["attention_mask"]}

    return ExampleInputs(
        status="ok",
        clean_enc=clean_enc, control_enc=control_enc, attack_enc=attack_enc,
        query_span=spans.query_span,
        doc_span_clean=spans.doc_span_original,
        doc_span_control_attack=spans.doc_span_control_attack,
        attack_span_indices=sorted(align_result.inserted_positions),
    )
