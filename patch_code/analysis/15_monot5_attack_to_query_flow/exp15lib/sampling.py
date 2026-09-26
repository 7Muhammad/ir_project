"""
exp15lib/sampling.py
=====================
Successful-instance filtering and the deterministic seed-42 sample.

Population (DECISIONS.md items 6-9, 24)
---------------------------------------
* Pool: Exp 01's cached per-attack selection
  `{exp1_attacks_dir}/{attack}/scores/selected_examples.jsonl` (reused, never
  rescored; carries query / passage / attacked_passage / control_score /
  attack_score).
* Success: delta = attack_score - control_score > SKIP_EPSILON, with
  SKIP_EPSILON imported from Exp 01 (src.patching). The stored
  attack_delta_vs_control must agree with the recomputed delta (fail loudly
  otherwise: it would mean the cache is internally inconsistent).
* Sample: shuffle the successful pool with `random.Random(f"{seed}:{attack}")`
  and take a prefix of length min(50, n_successful). This is exactly Exp 13's
  per-attack seeding scheme (exp13lib.run_utils.select_examples_for_attack),
  so Exp 15's sample for an attack is a superset of Exp 13's n<=50 prefix —
  but the success threshold here is the IMPORTED constant, not Exp 13's local
  copy.
"""

from __future__ import annotations

import json
import pathlib
import random
from typing import Dict, List

import exp15lib  # noqa: F401
from src.patching import SKIP_EPSILON

STORED_DELTA_ATOL = 1e-4


def load_pool(exp1_attacks_dir: pathlib.Path, attack_name: str) -> List[Dict]:
    path = pathlib.Path(exp1_attacks_dir) / attack_name / "scores" / "selected_examples.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Exp 01 cached pool missing for {attack_name}: {path}")
    with open(path, encoding="utf-8") as fh:
        pool = [json.loads(l) for l in fh if l.strip()]
    if not pool:
        raise ValueError(f"Exp 01 cached pool is empty: {path}")
    return pool


def successful_instances(pool: List[Dict]) -> List[Dict]:
    out = []
    for ex in pool:
        for key in ("qid", "docid", "query", "passage", "attacked_passage", "control_score", "attack_score"):
            if key not in ex:
                raise KeyError(f"Exp 01 pool record lacks '{key}': qid={ex.get('qid')} docid={ex.get('docid')}")
        delta = ex["attack_score"] - ex["control_score"]
        stored = ex.get("attack_delta_vs_control")
        if stored is not None and abs(stored - delta) > STORED_DELTA_ATOL:
            raise ValueError(
                f"Inconsistent cached pool record {ex['qid']}/{ex['docid']}: stored delta {stored} "
                f"!= attack_score - control_score = {delta}"
            )
        if delta > SKIP_EPSILON:
            out.append(ex)
    return out


def sample_attack(pool: List[Dict], attack_name: str, seed: int, max_n: int) -> List[Dict]:
    """Deterministic seed-keyed shuffle of the successful pool, then a prefix of <= max_n."""
    succ = successful_instances(pool)
    ids = [f"{e['qid']}_{e['docid']}" for e in succ]
    if len(set(ids)) != len(ids):
        raise ValueError(f"Duplicate (qid, docid) in Exp 01 pool for {attack_name}")
    order = list(range(len(succ)))
    random.Random(f"{seed}:{attack_name}").shuffle(order)
    return [succ[i] for i in order[:max_n]]
