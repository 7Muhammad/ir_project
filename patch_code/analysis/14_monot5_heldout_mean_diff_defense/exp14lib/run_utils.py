"""
exp14lib/run_utils.py
=======================
Config loading (reused from Experiment 3, which already generalized the
``attacks.inherit_from`` resolution — see headlib/run_utils.py) plus the
per-item status.json resume pattern, copied verbatim from Experiment 1's
scripts/10_run_multi_attack_pipeline.py (``_write_status`` /
``_is_already_successful``), which Experiments 6 and 12 already reuse the
same way.
"""

from __future__ import annotations

import json
import pathlib
from typing import List

from headlib.run_utils import load_config, resolve_cfg_path  # noqa: F401


def write_status(item_dir: pathlib.Path, status: dict) -> None:
    item_dir.mkdir(parents=True, exist_ok=True)
    status_path = item_dir / "status.json"
    with open(status_path, "w", encoding="utf-8") as fh:
        json.dump(status, fh, indent=2)


def is_already_successful(item_dir: pathlib.Path, required_files: List[str] = None) -> bool:
    """
    Return True if `item_dir/status.json` exists, has status=="success", and
    every file in `required_files` (relative to item_dir) exists — guards
    against mistaking a partial/corrupted prior run for a completed one.
    """
    status_path = item_dir / "status.json"
    if not status_path.exists():
        return False
    try:
        with open(status_path, encoding="utf-8") as fh:
            st = json.load(fh)
    except Exception:
        return False
    if st.get("status") != "success":
        return False
    for rel in (required_files or []):
        if not (item_dir / rel).exists():
            return False
    return True


def load_status(item_dir: pathlib.Path) -> dict:
    with open(item_dir / "status.json", encoding="utf-8") as fh:
        return json.load(fh)
