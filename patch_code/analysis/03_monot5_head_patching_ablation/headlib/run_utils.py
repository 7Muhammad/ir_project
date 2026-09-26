"""
headlib/run_utils.py
====================
Config loading and example selection for Experiment 3.

Example selection: reuse-first
-------------------------------
Experiment 1's multi-attack grid already scored all pairs of every attack and
saved up to 200 selected examples per attack, sorted by attack_delta_vs_control
descending (outputs/attacks/{attack}/scores/selected_examples.jsonl).  Each
record carries query, passage, attacked_passage and all three scores, which is
everything Experiment 3 needs.  Reusing them (a) avoids re-scoring 105 x 500
pairs and (b) guarantees Grid A/B run on exactly the examples Experiment 1
analysed.

If a selection file is missing (e.g. fresh checkout, smoke tests on new
attacks), we fall back to scoring + selecting with the same upstream logic
(src.scoring.score_pairs_general + select_attacked_examples).
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
    Load the experiment config, resolving the ``attacks.inherit_from``
    indirection.

    ``attacks.inherit_from`` points at Experiment 1's multi_attack.yaml; its
    ``attacks:`` block (the 7 tokens x 3 positions x 5 reps = 105-attack grid)
    is loaded as the base, then any non-null keys set locally override it.
    This reuses the attack-grid definition instead of copying 105 names.

    All relative paths in the config are interpreted relative to the config
    file's directory; this function resolves inherit_from that way.
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
        overrides = {
            k: v for k, v in attacks_cfg.items()
            if k != "inherit_from" and v is not None
        }
        cfg["attacks"] = {**base_attacks, **overrides}

    cfg["_config_dir"] = str(config_path.parent)
    return cfg


def resolve_cfg_path(cfg: dict, path_str: str) -> pathlib.Path:
    """Resolve a config-relative path string against the config's directory."""
    p = pathlib.Path(path_str)
    if p.is_absolute():
        return p
    return (pathlib.Path(cfg["_config_dir"]) / p).resolve()


# ---------------------------------------------------------------------------
# Example selection
# ---------------------------------------------------------------------------

def load_reused_examples(
    attack_name: str,
    n_examples: int,
    reuse_dir: pathlib.Path,
) -> Optional[List[Dict]]:
    """
    Load the top-n examples for one attack from Experiment 1's outputs.

    Returns None when the selection file does not exist.  Records are
    re-sorted by attack_delta_vs_control descending (defensive; the file is
    already written in that order) before taking the first n.
    """
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
    print(
        f"[run_utils] Reusing Experiment-1 selection for {attack_name}: "
        f"{min(n_examples, len(examples))}/{len(examples)} examples from {path}"
    )
    return examples[:n_examples]


def score_and_select_examples(
    spec: AttackSpec,
    n_examples: int,
    cfg: dict,
    model,
    tokenizer,
    true_id: int,
    false_id: int,
    device,
) -> List[Dict]:
    """
    Fallback: score the attacked TSV from scratch and select the top-n
    examples where the attack beats the padded control — the same logic
    Experiment 1's Stage 00/01 applies (via importable upstream functions).
    """
    data_cfg = cfg["data"]
    pairs = load_attacked_tsv(
        path=spec.path,
        max_pairs=data_cfg["max_pairs"],
        max_docs_per_query=data_cfg["max_docs_per_query"],
        seed=cfg["runtime"]["seed"],
    )
    scored, _n_align_failed = score_pairs_general(
        pairs=pairs,
        model=model,
        tokenizer=tokenizer,
        true_id=true_id,
        false_id=false_id,
        max_length=cfg["model"]["max_length"],
        device=device,
        batch_size=cfg["runtime"]["batch_size_scoring"],
    )
    return select_attacked_examples(
        scored_pairs=scored,
        min_attack_delta=cfg["selection"]["min_attack_delta"],
        max_selected=n_examples,
        seed=cfg["runtime"]["seed"],
    )


def get_examples_for_attack(
    spec: AttackSpec,
    n_examples: int,
    cfg: dict,
    model,
    tokenizer,
    true_id: int,
    false_id: int,
    device,
) -> List[Dict]:
    """Reuse Experiment 1's selection when available, else score + select."""
    reuse_dir_str = cfg["selection"].get("reuse_dir")
    if reuse_dir_str:
        reused = load_reused_examples(
            spec.attack_name, n_examples, resolve_cfg_path(cfg, reuse_dir_str)
        )
        if reused is not None:
            return reused
        print(
            f"[run_utils] No Experiment-1 selection for {spec.attack_name} — "
            "falling back to scoring from scratch."
        )
    return score_and_select_examples(
        spec, n_examples, cfg, model, tokenizer, true_id, false_id, device
    )
