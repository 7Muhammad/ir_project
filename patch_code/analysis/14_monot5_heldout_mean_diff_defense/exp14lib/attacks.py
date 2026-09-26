"""
exp14lib/attacks.py
=====================
Attack-grid helpers. Reuses Experiment 1's attack discovery
(src.attack_registry.discover_attacks) and the same
``attacks.inherit_from`` config indirection Experiments 2/3/6/etc. use to
pull in the 105-attack grid definition (7 tokens x 3 positions x 5 reps)
without copying the list — see exp14lib/run_utils.load_config, which
re-exports headlib.run_utils.load_config's inherit_from resolution.
"""

from __future__ import annotations

import re
from typing import List, Tuple

from src.attack_registry import AttackSpec, discover_attacks

_NAME_PATTERN = re.compile(r"^(?P<token>.+)_(?P<position>start|end|random)_(?P<reps>[1-9]\d*)$")


def parse_attack_name(attack_name: str) -> Tuple[str, str, int]:
    """Split a normalised attack_name ('relevant_start_5') into (token, position, repetitions)."""
    m = _NAME_PATTERN.match(attack_name)
    if m is None:
        raise ValueError(f"Cannot parse attack_name '{attack_name}' as '{{token}}_{{start|end|random}}_{{reps}}'.")
    return m.group("token"), m.group("position"), int(m.group("reps"))


def list_all_attacks(cfg: dict) -> List[AttackSpec]:
    """The full 105-attack grid (7 tokens x 3 positions x 5 reps), from Experiment 1's config."""
    return discover_attacks(cfg["attacks"])
