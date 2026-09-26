"""
exp2lib/run_utils.py
======================
Config loading and example/direction I/O glue for Experiment 2.

Config loading (``attacks.inherit_from`` resolution) and example reuse
(Experiment 1's ``selected_examples.jsonl``) are generic, experiment-agnostic
utilities already implemented for Experiment 3 — re-exported here rather than
duplicated (see headlib/run_utils.py).
"""

from __future__ import annotations

import csv
import pathlib
from typing import Dict, List, Optional, Tuple

import torch
from transformers import T5Tokenizer

from headlib.run_utils import load_config, load_reused_examples, resolve_cfg_path  # noqa: F401
from src.attack_registry import AttackSpec, discover_attacks
from src.model_utils import build_monot5_input

DirKey = Tuple[str, str, int, Optional[int]]


def attacks_for_tier(tier: str, cfg: dict) -> List[AttackSpec]:
    """
    grid_a sweeps the full inherited attack grid; grid_b uses the single
    canonical attack named in the config, resolved by name (works even when
    the grid is truncated by max_attacks in smoke configs) — same pattern as
    Experiment 3's attacks_for_run.
    """
    if tier == "grid_a":
        return discover_attacks(cfg["attacks"])
    canonical = cfg["tiers"]["grid_b"]["attack_name"]
    single_cfg = {
        "upstream_injected_dir": cfg["attacks"]["upstream_injected_dir"],
        "mode": "include",
        "include": [canonical],
    }
    return discover_attacks(single_cfg)


def n_examples_for_tier(tier: str, cfg: dict) -> int:
    if tier == "grid_a":
        return cfg["tiers"]["grid_a"]["n_examples_per_attack"]
    return cfg["tiers"]["grid_b"]["n_examples"]


def build_clean_encoding(
    tokenizer: T5Tokenizer,
    query: str,
    passage: str,
    max_length: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """
    Type-A clean encoding: plain text, normal tokenizer padding — no
    padded-control alignment needed since the sufficiency test only shifts
    one head's activation via a hook, not a prefix-position patch.
    """
    text = build_monot5_input(query, passage)
    tok = tokenizer(
        text, return_tensors="pt", max_length=max_length, truncation=True, padding=False,
    )
    return {
        "input_ids": tok["input_ids"].to(device),
        "attention_mask": tok["attention_mask"].to(device),
    }


def save_directions(directions: Dict[DirKey, torch.Tensor], path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(directions, path)


def load_directions(path: pathlib.Path) -> Dict[DirKey, torch.Tensor]:
    return torch.load(path, map_location="cpu", weights_only=True)


def save_norm_rows(rows: List[dict], path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["tier", "granularity", "component", "layer", "head_idx",
              "direction_norm", "n_control", "n_attack"]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_direction_norms(path: pathlib.Path) -> Dict[DirKey, float]:
    """Load a `{tier}_direction_norms.csv` into a DirKey -> norm lookup."""
    norms: Dict[DirKey, float] = {}
    with open(path, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            head_idx = int(row["head_idx"]) if row["head_idx"] != "" else None
            key = (row["granularity"], row["component"], int(row["layer"]), head_idx)
            norms[key] = float(row["direction_norm"])
    return norms
