"""
exp14lib/ood_folds.py
=======================
Leave-one-factor-value-out fold construction for attack-OOD evaluation
(task spec section 11): leave-one-token-out (7 folds), leave-one-position-
out (3 folds), leave-one-repetition-out (5 folds).

Each fold partitions the 105-attack grid into `seen_attacks` (used for
direction fitting and scale selection) and `held_out_attacks` (evaluated
only, never fit on) by one factor value. Test pairs are additionally
pair-held-out via the existing train/validation/test split — the fold only
controls which ATTACK CONFIGURATIONS are visible, independent of which
PAIRS are visible (task spec section 11: "the held-out attack factor is
also unseen during direction fitting").
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from src.attack_registry import AttackSpec

FACTORS = ["token", "position", "repetitions"]


@dataclass(frozen=True)
class OODFold:
    factor: str            # "token" | "position" | "repetitions"
    held_out_value: str    # e.g. "relevant", "start", "3" (repetitions stored as str for JSON round-trip)
    seen_attacks: List[str]
    held_out_attacks: List[str]


def build_folds(all_specs: List[AttackSpec], factor: str) -> List[OODFold]:
    if factor not in FACTORS:
        raise ValueError(f"Unknown OOD factor '{factor}'. Expected one of {FACTORS}.")

    values = sorted({str(getattr(s, factor)) for s in all_specs})
    folds: List[OODFold] = []
    for v in values:
        held_out = [s.attack_name for s in all_specs if str(getattr(s, factor)) == v]
        seen = [s.attack_name for s in all_specs if str(getattr(s, factor)) != v]
        folds.append(OODFold(factor=factor, held_out_value=v, seen_attacks=seen, held_out_attacks=held_out))
    return folds


def build_all_folds(all_specs: List[AttackSpec]) -> Dict[str, List[OODFold]]:
    return {factor: build_folds(all_specs, factor) for factor in FACTORS}


def folds_to_manifest(all_folds: Dict[str, List[OODFold]]) -> dict:
    return {
        factor: [
            {
                "factor": f.factor,
                "held_out_value": f.held_out_value,
                "n_seen_attacks": len(f.seen_attacks),
                "n_held_out_attacks": len(f.held_out_attacks),
                "seen_attacks": f.seen_attacks,
                "held_out_attacks": f.held_out_attacks,
            }
            for f in folds
        ]
        for factor, folds in all_folds.items()
    }


def manifest_to_folds(manifest: dict) -> Dict[str, List[OODFold]]:
    return {
        factor: [
            OODFold(
                factor=row["factor"], held_out_value=row["held_out_value"],
                seen_attacks=row["seen_attacks"], held_out_attacks=row["held_out_attacks"],
            )
            for row in rows
        ]
        for factor, rows in manifest.items()
    }
