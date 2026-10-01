"""
exp18lib/config.py
===================
YAML config with `extends:` (recursive merge, lists replaced) — the Exp 16 convention — and
path resolution relative to the config file. The model block of the Exp 16 config is reused
(checkpoint, max_length, fp32) so the forward pass is identical to Exp 16 stage 18.
"""

from __future__ import annotations

import copy
import pathlib

import yaml

import exp18lib  # noqa: F401
from exp16lib.config import load_config as load_exp16_config

EXP_DIR = exp18lib.EXP_DIR


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def load_config(path) -> dict:
    path = pathlib.Path(path).resolve()
    raw = yaml.safe_load(path.read_text())
    parent = raw.pop("extends", None)
    cfg = _merge(load_config(path.parent / parent), raw) if parent else raw
    cfg["_config_dir"] = str(path.parent)
    return cfg


def resolve(cfg: dict, rel: str) -> pathlib.Path:
    return (pathlib.Path(cfg["_config_dir"]) / rel).resolve()


def output_dir(cfg: dict) -> pathlib.Path:
    return resolve(cfg, cfg["outputs"]["base_dir"])


def model_config(cfg: dict) -> dict:
    """The Exp 16 config (model block reused) with this experiment's device override."""
    c16 = load_exp16_config(resolve(cfg, cfg["exp16_config"]))
    c16["model"]["device"] = cfg["model"]["device"]
    return c16
