"""
exp2lib/metrics.py
====================
Per-row intervention metrics.

2a defense: did subtracting the direction from the attacked score move it
closer to the control baseline than the unmodified attack score was?

2b sufficiency: did adding the direction to the clean score move it closer
to the attack baseline than the unmodified clean score was? (Mirrors 2a's
formula with before=score_clean, target=score_attack, after=score_modified.)
"""

from __future__ import annotations


def delta_toward_control(score_attack: float, score_control: float, score_modified: float) -> float:
    return abs(score_attack - score_control) - abs(score_modified - score_control)


def delta_toward_attack(score_clean: float, score_attack: float, score_modified: float) -> float:
    return abs(score_clean - score_attack) - abs(score_modified - score_attack)
