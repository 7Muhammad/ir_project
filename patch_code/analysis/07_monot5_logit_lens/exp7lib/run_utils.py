"""
exp7lib/run_utils.py
======================
Config loading, example reuse, per-example input assembly, and
Experiment-3 flagged-head loading for Experiment 7.
"""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
import yaml
from transformers import T5Tokenizer

from src.attack_registry import AttackSpec
from src.data_utils import load_attacked_tsv
from src.model_utils import build_monot5_input, build_padded_control_and_attack_encodings_general
from src.scoring import score_pairs_general, select_attacked_examples


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(config_path: pathlib.Path) -> dict:
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
# Flagged heads (Experiment 3's output, read live — not hardcoded)
# ---------------------------------------------------------------------------

def load_flagged_heads(cfg: dict) -> List[Tuple[int, int]]:
    """
    Read Experiment 3's aggregated head_summary.csv and return (layer, head)
    pairs for decoder_cross_attn heads exceeding the configured threshold.
    See DECISIONS.md for the default threshold (0.02) and its provenance.
    """
    fh_cfg = cfg["flagged_heads"]
    path = resolve_cfg_path(cfg, fh_cfg["source_csv"])
    if not path.exists():
        raise FileNotFoundError(
            f"Experiment 3 head summary not found at {path}. "
            "Run Experiment 3's grid_a + aggregate stages first."
        )
    df = pd.read_csv(path)
    df = df[df["component"] == "decoder_cross_attn"]
    threshold = fh_cfg["combined_effect_threshold"]
    flagged = df[df["combined_effect_mean"] > threshold].sort_values("combined_effect_mean", ascending=False)
    pairs = list(zip(flagged["layer"].astype(int).tolist(), flagged["head_idx"].astype(int).tolist()))
    print(f"[run_utils] {len(pairs)} flagged decoder_cross_attn heads "
          f"(combined_effect_mean > {threshold}) from {path}")
    return pairs


# ---------------------------------------------------------------------------
# Example reuse (same reuse-first policy as Experiments 3 / 6)
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
# Per-example input assembly: Type A/B/C encodings + attack-token id
# ---------------------------------------------------------------------------

@dataclass
class ExampleInputs:
    status: str                  # "ok" | "align_failed"
    reason: str = ""
    clean_enc: Optional[Dict[str, torch.Tensor]] = None       # Type A
    control_enc: Optional[Dict[str, torch.Tensor]] = None     # Type B
    attack_enc: Optional[Dict[str, torch.Tensor]] = None      # Type C
    attack_token_id: Optional[int] = None


def build_example_inputs(
    tokenizer: T5Tokenizer,
    example: Dict,
    attack_token_text: str,
    max_length: int,
    device: torch.device,
) -> ExampleInputs:
    """
    Build Type A (clean), Type B (padded control), Type C (attacked)
    encodings for one example, plus the attack token's single vocabulary id
    (all 7 attack-grid words are single SentencePiece tokens — verified;
    Analysis 5 tracks this one id's rank/logit at every layer).
    """
    query, passage, attacked_passage = example["query"], example["passage"], example["attacked_passage"]

    control_enc, attack_enc, align_result = build_padded_control_and_attack_encodings_general(
        tokenizer=tokenizer, query=query, passage=passage, attacked_passage=attacked_passage,
        max_length=max_length, device=device,
    )
    if align_result.status != "ok":
        return ExampleInputs(status="align_failed", reason=f"attack alignment: {align_result.reason}")

    clean_text = build_monot5_input(query, passage)
    clean_tok = tokenizer(clean_text, return_tensors="pt", max_length=max_length, truncation=True, padding=False)
    clean_enc = {"input_ids": clean_tok["input_ids"], "attention_mask": clean_tok["attention_mask"]}

    attack_ids = tokenizer.encode(attack_token_text, add_special_tokens=False)
    if len(attack_ids) != 1:
        return ExampleInputs(status="align_failed",
                              reason=f"attack token '{attack_token_text}' is not a single token: {attack_ids}")

    return ExampleInputs(
        status="ok", clean_enc=clean_enc, control_enc=control_enc, attack_enc=attack_enc,
        attack_token_id=attack_ids[0],
    )
