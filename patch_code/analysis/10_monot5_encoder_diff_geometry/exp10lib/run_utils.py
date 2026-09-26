"""
exp10lib/run_utils.py
======================
Config loading (with the same ``attacks.inherit_from`` indirection used by
Experiments 2/3/6/7), attack-strength selection (top-N by
Experiment 1's ``mean_attack_delta``), example reuse (Experiment 1's
``selected_examples.jsonl``), and per-example Type B/C input assembly.

Nothing here re-derives model loading, scoring, or padded-control
construction — those are imported directly from Experiment 1's ``src``
package (see scripts/*.py for the sys.path wiring).
"""

from __future__ import annotations

import csv
import json
import pathlib
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import yaml
from transformers import T5Tokenizer

from src.alignment import AlignmentResult, align_attack
from src.attack_registry import discover_attacks
from src.model_utils import build_monot5_input, build_padded_control_and_attack_encodings_general


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(config_path: pathlib.Path) -> dict:
    """Load a YAML config, resolving ``attacks.inherit_from`` if present."""
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
# Attack-strength selection (Experiment 1's cross-attack summary)
# ---------------------------------------------------------------------------

def select_attacks(cfg: dict) -> List[str]:
    """
    Rank attacks by Experiment 1's ``mean_attack_delta`` (descending) and
    return the top ``selection.n_attacks`` attack_names (or all rows in the
    summary CSV if ``n_attacks`` is null — the optional full-105-attack
    extension in the task spec).

    Reuses Experiment 1's already-computed cross-attack summary
    (``outputs/attack_comparison/summary.csv``, produced by its
    ``scripts/11_compare_attacks.py``) rather than recomputing mean attack
    deltas here. Rows are also cross-checked against the ``attacks.
    inherit_from`` curated-grid definition (the same 105-attack grid used
    by Experiments 2/3/6/7) so a stale or hand-edited summary CSV can't
    silently select an attack outside that grid.
    """
    sel_cfg = cfg["selection"]
    summary_path = resolve_cfg_path(cfg, sel_cfg["summary_csv"])
    if not summary_path.exists():
        raise FileNotFoundError(
            f"Experiment 1 cross-attack summary not found at {summary_path}. "
            "Run Experiment 1's multi-attack grid + scripts/11_compare_attacks.py first."
        )
    valid_names = {spec.attack_name for spec in discover_attacks(cfg["attacks"])}

    rows: List[Dict] = []
    with open(summary_path, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["attack_name"] not in valid_names:
                continue
            row["mean_attack_delta"] = float(row["mean_attack_delta"])
            rows.append(row)
    rows.sort(key=lambda r: r["mean_attack_delta"], reverse=True)

    n_attacks: Optional[int] = sel_cfg.get("n_attacks")
    if n_attacks is not None:
        rows = rows[:n_attacks]

    names = [r["attack_name"] for r in rows]
    print(f"[run_utils] Selected {len(names)}/{len(valid_names)} attack(s) in-grid, "
          f"by mean_attack_delta (top: {names[0] if names else 'none'}).")
    return names


# ---------------------------------------------------------------------------
# Example reuse (Experiment 1's per-attack selected_examples.jsonl)
# ---------------------------------------------------------------------------

def load_reused_examples(attack_name: str, n_examples: int, reuse_dir: pathlib.Path) -> List[Dict]:
    """
    Load and re-truncate Experiment 1's curated example pool for one attack.

    Experiment 1 already sorts by ``attack_delta_vs_control`` descending and
    caps at ``max_selected_examples`` (200); we re-sort defensively (the
    ordering is a documented invariant of that file, but re-sorting here
    costs nothing and removes a hidden cross-experiment coupling) and take
    the top ``n_examples``.
    """
    path = reuse_dir / attack_name / "scores" / "selected_examples.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"No selected_examples.jsonl for attack '{attack_name}' at {path}. "
            "Experiment 10 reuses Experiment 1's curated example pool and does "
            "not re-score pairs itself; run Experiment 1's multi-attack grid first."
        )
    examples: List[Dict] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    examples.sort(key=lambda x: x["attack_delta_vs_control"], reverse=True)
    return examples[:n_examples]


# ---------------------------------------------------------------------------
# Per-example Type B/C input assembly (+ alignment result, for the exact
# injected-token sub-span — Experiment 6 does not expose this, see
# DECISIONS.md)
# ---------------------------------------------------------------------------

@dataclass
class ExampleInputs:
    status: str                                            # "ok" | "align_failed" | "multi_token_attack"
    reason: str = ""
    control_enc: Optional[Dict[str, torch.Tensor]] = None   # Type B
    attack_enc: Optional[Dict[str, torch.Tensor]] = None    # Type C
    align_result: Optional[AlignmentResult] = None
    seq_len: int = 0


def build_example_inputs(
    tokenizer: T5Tokenizer,
    example: Dict,
    max_length: int,
    device: torch.device,
) -> ExampleInputs:
    """
    Build Type B (padded control) and Type C (attacked) encodings for one
    example, plus the token-level alignment result that identifies exactly
    which positions in Type B/C are the injected attack tokens.
    """
    query, passage, attacked_passage = example["query"], example["passage"], example["attacked_passage"]

    control_enc, attack_enc, align_result = build_padded_control_and_attack_encodings_general(
        tokenizer=tokenizer, query=query, passage=passage, attacked_passage=attacked_passage,
        max_length=max_length, device=device,
    )
    if align_result.status != "ok":
        return ExampleInputs(status="align_failed", reason=f"attack alignment: {align_result.reason}",
                              align_result=align_result)

    seq_len = attack_enc["input_ids"].shape[1]
    return ExampleInputs(
        status="ok", control_enc=control_enc, attack_enc=attack_enc,
        align_result=align_result, seq_len=seq_len,
    )
