"""
exp13lib/run_utils.py
=======================
Config loading, model loading, and example selection for Experiment 13.

Example selection differs from Experiments 3/11's "top-N by stored delta"
policy: the task spec calls for a SEED-42 RANDOM sample of up to N
successful examples per attack (not necessarily the N most extreme
ones), with N configurable (10 for the initial breadth run, later 30 /
100) without touching implementation code. To make n=30 a superset-
consistent extension of n=10 (same examples reused, not re-shuffled), we
shuffle each attack's qualifying pool ONCE with a per-attack-seeded RNG
and take a prefix of that fixed order.

"Successful instance" filtering: examples are pooled from Experiment 1's
existing per-attack selection (reused, not rescored from scratch, per
project convention -- see headlib/run_utils.py's identical reuse-first
policy), pre-filtered on the STORED attack_delta_vs_control > 1e-4. The
actual delta used for path_forward/path_reverse normalization is always
the LIVE score_control/score_attack computed inside
path_engine.run_example_paths (from this experiment's own padded-control
encoding), which is asserted again to satisfy delta > 1e-4 (see
scripts/01_sanity_checks.py, check 6) -- the stored value is only used to
build a good candidate pool cheaply, never trusted for the final numbers.
"""

from __future__ import annotations

import json
import pathlib
import random
from typing import Dict, List, Optional

import torch
import yaml

EXP_DIR = pathlib.Path(__file__).parent.parent
EXP1_DIR = EXP_DIR.parent / "01_monot5_layer_patching"

import sys  # noqa: E402

for _p in (EXP1_DIR, EXP_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from src.attack_registry import AttackSpec, discover_attacks  # noqa: E402
from src.model_utils import (  # noqa: E402
    build_padded_control_and_attack_encodings_general,
    get_true_false_token_ids,
    load_monot5,
    resolve_device,
)

SKIP_EPSILON = 1e-4


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(config_path: pathlib.Path) -> dict:
    """Load the experiment config, resolving `attacks.inherit_from` (105-attack grid)."""
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


def get_attacks(cfg: dict) -> List[AttackSpec]:
    specs = discover_attacks(cfg["attacks"])
    subset = cfg.get("run", {}).get("attack_subset")
    if subset:
        by_name = {s.attack_name: s for s in specs}
        missing = [n for n in subset if n not in by_name]
        if missing:
            raise ValueError(f"run.attack_subset names not found: {missing}")
        specs = [by_name[n] for n in subset]
    return specs


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(cfg: dict):
    device = resolve_device(cfg["model"]["device"])
    model, tokenizer = load_monot5(cfg["model"]["checkpoint"], device)
    true_id, false_id = get_true_false_token_ids(tokenizer)
    return model, tokenizer, true_id, false_id, device


# ---------------------------------------------------------------------------
# Example selection: reuse Experiment 1's pool, filter delta > 1e-4,
# seed-42 shuffle, take a size-N prefix (N configurable, nested across N).
# ---------------------------------------------------------------------------

def load_candidate_pool(attack_name: str, reuse_dir: pathlib.Path) -> List[Dict]:
    """
    All of Experiment 1's stored selected examples for one attack (already
    filtered to attack_delta_vs_control > 0 at Experiment-1 selection
    time; up to ~200 per attack). Returns [] if the file is missing.
    """
    path = reuse_dir / attack_name / "scores" / "selected_examples.jsonl"
    if not path.exists():
        return []
    examples: List[Dict] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def select_examples_for_attack(
    attack_name: str,
    reuse_dir: pathlib.Path,
    max_examples: int,
    seed: int,
) -> List[Dict]:
    """
    Seed-42 shuffle of the attack's qualifying candidate pool (stored
    attack_delta_vs_control > SKIP_EPSILON), then take the first
    `max_examples`. The per-attack RNG is seeded from (seed, attack_name)
    so the shuffled order is stable across runs and across different
    values of `max_examples` -- an n=30 run's examples are exactly an
    n=10 run's 10 examples plus 20 more, not a different random draw.
    """
    pool = load_candidate_pool(attack_name, reuse_dir)
    qualifying = [p for p in pool if p.get("attack_delta_vs_control", 0.0) > SKIP_EPSILON]

    rng = random.Random(f"{seed}:{attack_name}")
    order = list(range(len(qualifying)))
    rng.shuffle(order)
    shuffled = [qualifying[i] for i in order]

    selected = shuffled[:max_examples]
    for i, ex in enumerate(selected):
        ex["sample_index"] = i
        ex["seed"] = seed
    return selected


# ---------------------------------------------------------------------------
# Encoding construction (position-agnostic; reused for start/end/random)
# ---------------------------------------------------------------------------

def build_encodings(tokenizer, example: Dict, max_length: int, device: torch.device):
    """
    Build (control_enc, attack_enc) for one example using the project's
    general position-agnostic alignment (src.model_utils
    build_padded_control_and_attack_encodings_general), which supports
    start/end/random injection attacks alike. Returns
    (control_enc, attack_enc, align_result); check align_result.status.
    """
    return build_padded_control_and_attack_encodings_general(
        tokenizer=tokenizer,
        query=example["query"],
        passage=example["passage"],
        attacked_passage=example["attacked_passage"],
        max_length=max_length,
        device=device,
    )
