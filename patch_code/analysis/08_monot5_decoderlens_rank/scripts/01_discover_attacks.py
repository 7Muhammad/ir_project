#!/usr/bin/env python3
"""
scripts/01_discover_attacks.py
==============================
Discover all available Parry / ECIR-24 manual injection attacks and write a
metadata table.  Uses the attack registry with the config's ``attacks`` block
(default mode "all").

Output
------
  outputs/discovered_attacks.csv

Columns: attack_name, token, position, repetitions, run_name, path

Usage
-----
  python scripts/01_discover_attacks.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import sys

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.attack_registry import discover_attacks
from src.utils import load_config


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Discover ECIR-24 injection attacks.")
    p.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "default.yaml"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    attacks = discover_attacks(cfg["attacks"])

    base_dir = PROJECT_ROOT / cfg["outputs"]["base_dir"]
    base_dir.mkdir(parents=True, exist_ok=True)
    out_path = base_dir / "discovered_attacks.csv"

    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["attack_name", "token", "position", "repetitions", "run_name", "path"]
        )
        for spec in attacks:
            writer.writerow(
                [spec.attack_name, spec.token, spec.position,
                 spec.repetitions, spec.run_name, str(spec.path)]
            )

    print(f"[01] Discovered {len(attacks)} attacks.")
    print(f"[01] Wrote: {out_path}")
    # Print a small preview.
    for spec in attacks[:10]:
        print(f"     {spec.attack_name:30s} token={spec.token!r:20s} "
              f"pos={spec.position:6s} reps={spec.repetitions} run={spec.run_name}")
    if len(attacks) > 10:
        print(f"     ... and {len(attacks) - 10} more")


if __name__ == "__main__":
    main()
