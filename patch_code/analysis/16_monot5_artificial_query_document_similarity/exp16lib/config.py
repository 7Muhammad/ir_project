"""
exp16lib/config.py
===================
Config loading. Reuses Experiment 3's `headlib.run_utils.load_config`, which
resolves `attacks.inherit_from` (the 105-attack grid in Exp 01's
multi_attack.yaml, never copied), plus `extends: <other.yaml>` so smoke.yaml
inherits default.yaml and overrides only what it sets (recursive merge;
lists are replaced, not concatenated).
"""

from __future__ import annotations

import copy
import pathlib
from typing import List

import yaml

import exp16lib  # noqa: F401  (sibling sys.path setup)
from headlib.run_utils import load_config as _load_config_with_attack_inheritance
from headlib.run_utils import resolve_cfg_path  # noqa: F401  (re-exported)
from src.attack_registry import AttackSpec, discover_attacks


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(config_path) -> dict:
    config_path = pathlib.Path(config_path).resolve()
    with open(config_path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    parent = raw.pop("extends", None)
    if parent is None:
        cfg = _load_config_with_attack_inheritance(config_path)
        inherit = (raw.get("attacks") or {}).get("inherit_from")
        cfg["_attack_grid_source"] = str((config_path.parent / inherit).resolve()) if inherit else str(config_path)
    else:
        base = load_config(config_path.parent / parent)
        base.pop("_config_dir", None)
        base.pop("_config_path", None)
        cfg = _deep_merge(base, raw)
    cfg["_config_dir"] = str(config_path.parent)
    cfg["_config_path"] = str(config_path)
    _validate(cfg)
    return cfg


def _validate(cfg: dict) -> None:
    q = cfg["qrels"]
    if sorted(q["relevant_grades"]) != [2, 3] or q["nonrelevant_grades"] != [0] or q["excluded_grades"] != [1]:
        raise ValueError("qrel groups are locked: relevant={2,3}, nonrelevant={0}, excluded={1} (DECISIONS 11-13)")
    if int(q["seed"]) != 42:
        raise ValueError("qrel balancing seed is locked to 42 (DECISIONS 16)")
    if cfg["model"].get("dtype", "float32") != "float32":
        raise ValueError("Exp 16 runs the model in float32 (cached Exp 01 scores are fp32)")


def output_dir(cfg: dict) -> pathlib.Path:
    return resolve_cfg_path(cfg, cfg["outputs"]["base_dir"])


def get_attacks(cfg: dict) -> List[AttackSpec]:
    """Attack specs in grid order (Exp 01 registry; fails loudly on missing TSVs)."""
    return discover_attacks(cfg["attacks"])
