#!/usr/bin/env python3
"""
scripts/02_prepare_decoderlens_pairs.py
=======================================
For an attack, build the per-query candidate sets and select the target
examples, running token-level alignment so each target has aligned
original / padded_control / attack variants.

Selection (config ``selection.mode``):
  - all_aligned (default): keep every example whose alignment succeeds, capped
    at max_target_examples_per_attack.  Weak / negative attacks are KEPT — that
    is intentional, we want to compare their layerwise rank curves too.
  - final_layer_success / top_delta: a larger aligned pool is kept here and the
    final selection is applied after scoring in Stage 03.

Alignment failures are reported (never silently skipped):
    n_alignment_success, n_alignment_failed, n_examples_used.

Output
------
  outputs/attacks/{attack_name}/pairs/target_pairs.jsonl
  outputs/attacks/{attack_name}/pairs/alignment_report.json

Usage
-----
  python scripts/02_prepare_decoderlens_pairs.py --config configs/default.yaml \
      --attack-name relevant_start_5
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Dict, List, Optional

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.attack_registry import AttackSpec, discover_attacks
from src.data_loading import build_candidate_sets, iter_target_candidates, load_attack_tsv
from src.decoderlens import load_monot5, resolve_device
from src.padded_control import build_variant_encodings
from src.utils import attack_output_dirs, load_config


def prepare_pairs_for_attack(
    spec: AttackSpec,
    cfg: dict,
    tokenizer,
    device: torch.device,
    base_dir: pathlib.Path,
) -> Dict:
    """
    Build and write target_pairs.jsonl for one attack.  Returns a counts dict.
    """
    out_dirs = attack_output_dirs(base_dir, spec.attack_name)
    candidate_top_k = int(cfg["ranking"]["candidate_top_k"])
    max_length = int(cfg["model"]["max_length"])
    sel = cfg["selection"]
    mode = sel.get("mode", "all_aligned")
    max_targets = int(sel.get("max_target_examples_per_attack", 100))

    # For modes that filter after scoring, keep a larger aligned pool.
    pool_cap = max_targets if mode == "all_aligned" else max_targets * 3

    records = load_attack_tsv(spec.path)
    candidate_sets = build_candidate_sets(records, candidate_top_k)
    flattened = iter_target_candidates(candidate_sets)

    n_align_ok = 0
    n_align_failed = 0
    kept: List[Dict] = []

    for cand in flattened:
        if len(kept) >= pool_cap:
            break
        _orig, _ctrl, _atk, result = build_variant_encodings(
            tokenizer=tokenizer,
            query=cand.query,
            passage=cand.passage,
            attacked_passage=cand.attacked_passage,
            max_length=max_length,
            device=device,
        )
        if result.status != "ok":
            n_align_failed += 1
            continue
        n_align_ok += 1
        kept.append(
            {
                "attack_name": spec.attack_name,
                "token": spec.token,
                "position": spec.position,
                "repetitions": spec.repetitions,
                "qid": cand.qid,
                "docid": cand.docid,
                "rank": cand.rank,
                "query": cand.query,
                "passage": cand.passage,
                "attacked_passage": cand.attacked_passage,
                "align_status": result.status,
                "n_inserted": result.n_inserted,
            }
        )

    pairs_path = out_dirs["pairs"] / "target_pairs.jsonl"
    with open(pairs_path, "w", encoding="utf-8") as fh:
        for rec in kept:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    report = {
        "attack_name": spec.attack_name,
        "candidate_top_k": candidate_top_k,
        "n_queries": len(candidate_sets),
        "n_alignment_success": n_align_ok,
        "n_alignment_failed": n_align_failed,
        "n_examples_used": len(kept),
        "selection_mode": mode,
    }
    with open(out_dirs["pairs"] / "alignment_report.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    print(
        f"[02] {spec.attack_name}: kept {len(kept)} aligned targets "
        f"(ok={n_align_ok}, failed={n_align_failed}) across "
        f"{len(candidate_sets)} queries -> {pairs_path}"
    )
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare DecoderLens target pairs.")
    p.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "default.yaml"))
    p.add_argument("--attack-name", default=None,
                   help="Single attack to prepare (default: all discovered).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    device = resolve_device(cfg["model"]["device"])
    # Tokeniser only (no model forward needed for alignment).
    _model, tokenizer = load_monot5(cfg["model"]["checkpoint"], device)
    base_dir = PROJECT_ROOT / cfg["outputs"]["base_dir"]

    attacks = discover_attacks(cfg["attacks"])
    if args.attack_name:
        attacks = [a for a in attacks if a.attack_name == args.attack_name]
        if not attacks:
            raise SystemExit(f"Attack '{args.attack_name}' not found.")

    for spec in attacks:
        prepare_pairs_for_attack(spec, cfg, tokenizer, device, base_dir)


if __name__ == "__main__":
    main()
