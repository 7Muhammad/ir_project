"""
exp14lib/data_pool.py
=======================
Successful-attack-instance loading, restricted by (pre-computed, pair-level)
split membership.

Reuses Experiment 1's per-attack selection output
(outputs/attacks/{attack_name}/scores/selected_examples.jsonl — up to 200
examples per attack where attack_delta_vs_control > 0, sorted descending)
instead of rescoring from scratch. Each record already carries query,
passage, attacked_passage, original_score (Type-A clean), control_score,
attack_score — everything downstream stages need (see Experiment 3's
headlib/run_utils.py, which established this reuse-first pattern).

Success criterion (task spec section 4): score_attack - score_control >
SUCCESS_THRESHOLD (1e-4, matching src.patching.SKIP_EPSILON's degenerate-
example guard used throughout the project). Experiment 1's own selection
threshold is looser (> 0.0), so we re-filter client-side rather than assume
its output already matches — no rescoring needed, this is a free filter on
already-cached scores.
"""

from __future__ import annotations

import json
import pathlib
from typing import Dict, List, Optional, Tuple

SUCCESS_THRESHOLD = 1e-4


def _selection_path(attack_name: str, exp1_attacks_dir: pathlib.Path) -> pathlib.Path:
    return exp1_attacks_dir / attack_name / "scores" / "selected_examples.jsonl"


def load_attack_examples(
    attack_name: str,
    exp1_attacks_dir: pathlib.Path,
    pair_to_split: Dict[Tuple[str, str], str],
    split_name: str,
    cap: Optional[int] = None,
) -> List[Dict]:
    """
    Load successful instances of one attack, restricted to pairs assigned to
    `split_name`. Records are already sorted by attack_delta_vs_control
    descending (Experiment 1's convention); `cap` keeps the top-`cap` after
    the split/threshold filter, if set.

    Returns an empty list (not an error) if the attack's selection file is
    missing or has zero examples in this split — this is expected for the
    ~93/105 attacks whose mean effect is negative (task section 4: a
    negative-mean attack can still contain individual successful instances,
    or none at all in a given split).
    """
    path = _selection_path(attack_name, exp1_attacks_dir)
    if not path.exists():
        return []

    records: List[Dict] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            key = (str(r["qid"]), str(r["docid"]))
            if pair_to_split.get(key) != split_name:
                continue
            if r["attack_delta_vs_control"] <= SUCCESS_THRESHOLD:
                continue
            records.append(r)

    records.sort(key=lambda x: x["attack_delta_vs_control"], reverse=True)
    if cap is not None:
        records = records[:cap]
    return records


def collect_clean_examples_for_split(
    attack_names: List[str],
    exp1_attacks_dir: pathlib.Path,
    pair_to_split: Dict[Tuple[str, str], str],
    split_name: str,
    cap: Optional[int] = None,
) -> List[Dict]:
    """
    One representative clean (Type-A) record per unique pair in `split_name`
    that appears in at least one attack's successful-instance pool.

    query/passage/original_score are attack-independent (same underlying
    document; monoT5 scores the SAME clean prompt regardless of which
    attack's TSV we happened to read it from), so the first record found for
    a given pair is kept and later duplicates from other attacks are
    skipped — this is deliberately NOT "one clean example per attack".
    """
    seen: Dict[Tuple[str, str], Dict] = {}
    for attack_name in attack_names:
        path = _selection_path(attack_name, exp1_attacks_dir)
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                key = (str(r["qid"]), str(r["docid"]))
                if pair_to_split.get(key) != split_name:
                    continue
                if key in seen:
                    continue
                seen[key] = {
                    "qid": r["qid"], "docid": r["docid"],
                    "query": r["query"], "passage": r["passage"],
                    "original_score": r["original_score"],
                }
    records = list(seen.values())
    records.sort(key=lambda x: (x["qid"], x["docid"]))
    if cap is not None:
        records = records[:cap]
    return records


def pool_round_robin(per_attack_examples: Dict[str, List[Dict]], cap: Optional[int]) -> List[Dict]:
    """
    Interleave examples across attacks (one at a time, highest-delta-first
    within each attack, attacks visited in sorted-name order) so a capped
    pool represents many attacks rather than being dominated by whichever
    single attack has the most successful instances. Deterministic given
    the same input dict. Not prescribed by the task spec — see DECISIONS.md
    for why this policy was chosen over a flat sort-by-delta pool.
    """
    attack_names = sorted(per_attack_examples.keys())
    cursors = {name: 0 for name in attack_names}
    pooled: List[Dict] = []
    progressed = True
    while progressed and (cap is None or len(pooled) < cap):
        progressed = False
        for name in attack_names:
            if cap is not None and len(pooled) >= cap:
                break
            examples = per_attack_examples[name]
            i = cursors[name]
            if i < len(examples):
                ex = dict(examples[i])
                ex["_source_attack"] = name
                pooled.append(ex)
                cursors[name] = i + 1
                progressed = True
    return pooled
