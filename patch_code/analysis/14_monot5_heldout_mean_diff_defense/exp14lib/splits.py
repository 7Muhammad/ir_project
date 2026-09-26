"""
exp14lib/splits.py
====================
Deterministic, pair-disjoint train/validation/test split.

The key methodological fix over Experiment 2 (see README "Motivation" and
DECISIONS.md item 2): Experiment 2 fit directions and tested interventions
on the SAME examples. Here, every split is done at the (qid, docid) PAIR
level, BEFORE any attack-success filtering, so that:

  - all 105 attack variants of a given pair land in the same split
    (a pair is never split across train/val/test), and
  - direction fitting only ever sees TRAIN pairs, scale selection only ever
    sees VALIDATION pairs, and held-out evaluation only ever sees TEST pairs.

The pair universe is Experiment 1's own canonical 500-pair sample
(outputs/pairs/pairs.jsonl), reused rather than resampled: every one of the
105 attack TSVs in the upstream ecir24 repo shares the IDENTICAL 42005-pair
BM25 candidate universe (verified empirically — see DECISIONS.md item 2),
so Experiment 1's fixed 500-pair sample is a valid, already-existing,
attack-independent pair pool to split on.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
from dataclasses import dataclass
from typing import Dict, List, Tuple

SPLIT_NAMES = ["train", "validation", "test"]
DEFAULT_RATIOS = {"train": 0.6, "validation": 0.2, "test": 0.2}


@dataclass(frozen=True)
class PairId:
    qid: str
    docid: str

    def key(self) -> str:
        return f"{self.qid}::{self.docid}"


def _stable_shuffle(pair_keys: List[str], seed: int) -> List[str]:
    """
    Deterministic, seed-controlled ordering that does not depend on Python's
    per-process hash randomization (unlike `hash()`), so the split is
    reproducible across processes/machines given the same seed and inputs.
    """
    def sort_key(k: str) -> str:
        return hashlib.sha256(f"{seed}::{k}".encode("utf-8")).hexdigest()

    return sorted(pair_keys, key=sort_key)


def build_split(
    pairs: List[Dict],
    seed: int = 42,
    ratios: Dict[str, float] = None,
) -> Dict[str, List[Dict]]:
    """
    Split `pairs` (each a dict with at least "qid", "docid") into
    train/validation/test by unique (qid, docid) pair.

    Ratios are applied to the number of UNIQUE pairs. Uses a fixed
    hash-based shuffle (see `_stable_shuffle`) so the split is exactly
    reproducible for a given seed regardless of input order.
    """
    ratios = ratios or DEFAULT_RATIOS
    if abs(sum(ratios.values()) - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {ratios} (sum={sum(ratios.values())})")

    by_key: Dict[str, Dict] = {}
    for p in pairs:
        pid = PairId(str(p["qid"]), str(p["docid"]))
        by_key.setdefault(pid.key(), p)  # first occurrence wins; pairs.jsonl has no dupes

    keys = _stable_shuffle(list(by_key.keys()), seed)
    n = len(keys)
    n_train = round(n * ratios["train"])
    n_val = round(n * ratios["validation"])
    # test gets the remainder so all pairs are assigned exactly once
    train_keys = keys[:n_train]
    val_keys = keys[n_train:n_train + n_val]
    test_keys = keys[n_train + n_val:]

    return {
        "train": [by_key[k] for k in train_keys],
        "validation": [by_key[k] for k in val_keys],
        "test": [by_key[k] for k in test_keys],
    }


def save_split_manifest(split: Dict[str, List[Dict]], seed: int, ratios: Dict[str, float], path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "seed": seed,
        "ratios": ratios,
        "n_pairs_total": sum(len(v) for v in split.values()),
        "splits": {
            name: [{"qid": str(p["qid"]), "docid": str(p["docid"])} for p in recs]
            for name, recs in split.items()
        },
    }
    assert_disjoint(manifest["splits"])
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)


def load_split_manifest(path: pathlib.Path) -> Dict[str, List[Tuple[str, str]]]:
    with open(path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    assert_disjoint(manifest["splits"])
    return {
        name: [(r["qid"], r["docid"]) for r in recs]
        for name, recs in manifest["splits"].items()
    }


def pair_to_split_map(splits: Dict[str, List[Tuple[str, str]]]) -> Dict[Tuple[str, str], str]:
    """Invert {split_name: [(qid, docid), ...]} into {(qid, docid): split_name}."""
    mapping: Dict[Tuple[str, str], str] = {}
    for name, pair_ids in splits.items():
        for pid in pair_ids:
            mapping[pid] = name
    return mapping


def assert_disjoint(splits: Dict[str, List[Dict]]) -> None:
    """Raise if any (qid, docid) pair appears in more than one split."""
    seen: Dict[str, str] = {}
    for split_name, recs in splits.items():
        for r in recs:
            key = f"{r['qid']}::{r['docid']}" if isinstance(r, dict) else f"{r[0]}::{r[1]}"
            if key in seen:
                raise ValueError(
                    f"Pair {key} appears in both split '{seen[key]}' and '{split_name}' — "
                    "splits must be disjoint."
                )
            seen[key] = split_name
