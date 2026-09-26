#!/usr/bin/env python3
"""
scripts/phase0_alignment_audit.py
===================================
Phase-0 real-data alignment audit.

Checks alignment success/failure rates on the first N rows of selected
attack TSV files from the upstream ECIR-24 repo, using the general
padded-control construction (build_padded_control_and_attack_encodings_general).

This script is purely diagnostic — it does not write any experiment outputs.

Output: per-attack table showing:
  attack_name     n_total   n_ok   n_failed   failure_rate   top_reasons (Counter)

Usage:
  conda activate advseq2seq
  python scripts/phase0_alignment_audit.py [--config configs/default.yaml] [--n 50]
"""

from __future__ import annotations

import argparse
import gzip
import pathlib
import sys
from collections import Counter
from typing import List

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import yaml
from transformers import T5Tokenizer

from src.model_utils import (
    build_padded_control_and_attack_encodings_general,
    build_monot5_input,
    load_monot5,
    resolve_device,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase-0 real-data alignment audit.")
    p.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "default.yaml"),
        help="Path to the YAML config file.",
    )
    p.add_argument(
        "--n",
        type=int,
        default=50,
        help="Number of rows to audit per attack (default: 50).",
    )
    p.add_argument(
        "--attacks",
        nargs="+",
        default=[
            "relevant_start_5",
            "relevant_end_5",
            "relevant_random_5",
            "information_start_5",
            "true_start_5",
            "false_start_5",
        ],
        help="Attack names to audit (default: 6 representative variants).",
    )
    return p.parse_args()


def _load_tsv_rows(path: pathlib.Path, n: int) -> List[dict]:
    """Load up to n rows from a .gz.tsv attacked file."""
    import csv
    rows = []
    opener = gzip.open if path.suffix == ".gz" or path.name.endswith(".gz.tsv") else open
    # Handle double-extension .gz.tsv
    if path.name.endswith(".gz.tsv"):
        opener = gzip.open
    with opener(path, "rt", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            rows.append(row)
            if len(rows) >= n:
                break
    return rows


def _audit_attack(
    attack_name: str,
    injected_dir: pathlib.Path,
    tokenizer: T5Tokenizer,
    device: torch.device,
    n: int,
) -> None:
    # Find the TSV file (any run, e.g. bm25_19)
    matches = list(injected_dir.glob(f"{attack_name}_*.gz.tsv"))
    if not matches:
        print(f"  {attack_name:40s}  FILE NOT FOUND in {injected_dir}")
        return

    tsv_path = matches[0]
    rows = _load_tsv_rows(tsv_path, n)

    n_ok      = 0
    n_failed  = 0
    reasons   = Counter()

    for row in rows:
        query            = row.get("query", "")
        passage          = row.get("text_0", "")
        attacked_passage = row.get("text", "")

        if not query or not passage or not attacked_passage:
            n_failed += 1
            reasons["missing_fields"] += 1
            continue

        _, _, result = build_padded_control_and_attack_encodings_general(
            tokenizer=tokenizer,
            query=query,
            passage=passage,
            attacked_passage=attacked_passage,
            max_length=512,
            device=device,
        )

        if result.status == "ok":
            n_ok += 1
        else:
            n_failed += 1
            # Truncate reason for display
            short = result.reason[:80].split("\n")[0]
            reasons[short] += 1

    total       = n_ok + n_failed
    fail_rate   = 100 * n_failed / total if total else 0
    ok_icon     = "✓" if fail_rate < 5 else ("!" if fail_rate < 20 else "✗")

    print(
        f"  {ok_icon} {attack_name:40s}  "
        f"total={total:4d}  ok={n_ok:4d}  failed={n_failed:3d}  "
        f"fail%={fail_rate:5.1f}%"
    )
    if n_failed > 0:
        for reason, cnt in reasons.most_common(3):
            print(f"      [{cnt:2d}x] {reason}")


def main() -> None:
    args = parse_args()
    cfg_path = pathlib.Path(args.config).resolve()

    with open(cfg_path) as fh:
        cfg = yaml.safe_load(fh)

    attacks_cfg  = cfg.get("attacks", {})
    injected_dir = pathlib.Path(
        attacks_cfg.get("upstream_injected_dir", "")
        or cfg.get("attack", {}).get("attacked_pairs_path", "")
    )
    # Resolve to directory if it points at a file
    if injected_dir.is_file():
        injected_dir = injected_dir.parent

    if not injected_dir.is_dir():
        sys.exit(
            f"ERROR: Could not locate injected directory.\n"
            f"Set attacks.upstream_injected_dir in {cfg_path}."
        )

    device = resolve_device(cfg["model"]["device"])

    print(f"[phase0_alignment_audit] Loading tokenizer ...")
    # Load tokenizer only (no full model needed for alignment check)
    from transformers import T5Tokenizer
    tokenizer = T5Tokenizer.from_pretrained(cfg["model"]["checkpoint"])

    print(f"[phase0_alignment_audit] Auditing {len(args.attacks)} attacks, {args.n} rows each.")
    print(f"[phase0_alignment_audit] Injected dir: {injected_dir}\n")
    print(f"  {'Attack':42s}  {'Results'}")
    print(f"  {'-'*42}  {'-'*55}")

    for attack_name in args.attacks:
        _audit_attack(attack_name, injected_dir, tokenizer, device, args.n)

    print(f"\n[phase0_alignment_audit] Done.")
    print("  ✓ < 5% failure rate: proceed to Phase 1")
    print("  ! 5–20% failure rate: investigate but continue (failures reported in status.json)")
    print("  ✗ > 20% failure rate: review alignment logic for this attack type")


if __name__ == "__main__":
    main()
