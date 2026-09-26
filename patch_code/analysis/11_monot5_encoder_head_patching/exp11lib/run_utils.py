"""
exp11lib/run_utils.py
=======================
Config loading and example selection for Experiment 11. Mirrors Experiment
3's headlib/run_utils.py and Experiment 6's exp6lib/run_utils.py
reuse-first design, written fresh here per project convention (each
experiment folder is self-contained; no experiment imports another
experiment's lib package except where a prompt explicitly calls for reuse
-- see exp11lib/spans_reuse.py for the one deliberate exception, Experiment
6's query/document span-finding logic).
"""

from __future__ import annotations

import json
import pathlib
from typing import Dict, List, Optional

import yaml

from src.attack_registry import AttackSpec
from src.data_utils import load_attacked_tsv
from src.scoring import score_pairs_general, select_attacked_examples


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(config_path: pathlib.Path) -> dict:
    """
    Load the experiment config, resolving ``attacks.inherit_from`` (points
    at Experiment 1's multi_attack.yaml 105-attack grid).
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
# Example selection (reuse-first, same policy as Experiments 3 and 6)
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
