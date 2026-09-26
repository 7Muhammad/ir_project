#!/usr/bin/env python3
"""
scripts/00_derive_candidate_heads.py
=======================================
Derive Experiment 14's candidate single-head list from Experiment 13's
already-vetted, checked-in head lists (configs/heads/encoder_senders.json,
decoder_receivers.json). Does NOT re-derive from Experiment 3/11's raw
aggregates — Experiment 13 already did that derivation and reconciliation
work (see exp13lib/head_lists.py's provenance note and this experiment's
DECISIONS.md item 1).

Asserts the expected counts (18 encoder, 31 decoder) and fails loudly,
rather than guessing, if they don't match.

Output: outputs/candidate_heads.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
EXP_DIR = SCRIPT_DIR.parent
EXP13_DIR = EXP_DIR.parent / "13_monot5_encoder_decoder_path_patching"
sys.path.insert(0, str(EXP13_DIR))
sys.path.insert(0, str(EXP_DIR))

from exp14lib.head_lists import derive_candidate_heads  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Derive Experiment 14's candidate head manifest.")
    p.add_argument("--out", default=str(EXP_DIR / "outputs" / "candidate_heads.json"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    exp13_heads_dir = EXP13_DIR / "configs" / "heads"
    manifest = derive_candidate_heads(exp13_heads_dir)

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    print(f"[00_derive_candidate_heads] {manifest['n_encoder_heads']} encoder heads "
          f"x {len(manifest['encoder_position_masks'])} position masks = "
          f"{manifest['n_encoder_candidates']} encoder candidates")
    print(f"[00_derive_candidate_heads] {manifest['n_decoder_heads']} decoder heads "
          f"= {manifest['n_decoder_candidates']} decoder candidates")
    print(f"[00_derive_candidate_heads] {manifest['n_total_candidates']} total candidate interventions "
          f"-> {out_path}")


if __name__ == "__main__":
    main()
