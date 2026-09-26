"""
exp15lib/config.py
===================
Config loading. Reuses Experiment 3's `headlib.run_utils.load_config`, which
resolves `attacks.inherit_from` (the 105-attack grid in Exp 01's
multi_attack.yaml), and adds one Exp-15 feature: a config may declare
`extends: <other.yaml>` to inherit every key from another config and
override only what it sets (smoke.yaml extends default.yaml). Overrides are
merged recursively; lists are replaced, not concatenated.
"""

from __future__ import annotations

import copy
import pathlib
from typing import List

import yaml

import exp15lib  # noqa: F401  (sibling sys.path setup)
from headlib.run_utils import load_config as _load_config_with_attack_inheritance
from headlib.run_utils import resolve_cfg_path  # noqa: F401  (re-exported)
from src.attack_registry import AttackSpec, discover_attacks
from src.patching import SKIP_EPSILON


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(config_path) -> dict:
    """Load a config, resolving `extends` (Exp 15) then `attacks.inherit_from` (Exp 3)."""
    config_path = pathlib.Path(config_path).resolve()
    with open(config_path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    parent = raw.pop("extends", None)
    if parent is None:
        cfg = _load_config_with_attack_inheritance(config_path)
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
    src = cfg["sampling"].get("success_epsilon_source")
    if src != "src.patching.SKIP_EPSILON":
        raise ValueError(
            f"sampling.success_epsilon_source must be 'src.patching.SKIP_EPSILON' (got {src!r}); "
            "Exp 15 never defines its own success epsilon (DECISIONS.md item 6)."
        )
    assert SKIP_EPSILON == 1e-4, f"Unexpected project SKIP_EPSILON={SKIP_EPSILON}"


def output_dir(cfg: dict) -> pathlib.Path:
    return resolve_cfg_path(cfg, cfg["outputs"]["base_dir"])


def get_attacks(cfg: dict) -> List[AttackSpec]:
    """Attack specs in grid order (Exp 01 registry; fails loudly on missing TSVs)."""
    return discover_attacks(cfg["attacks"])
