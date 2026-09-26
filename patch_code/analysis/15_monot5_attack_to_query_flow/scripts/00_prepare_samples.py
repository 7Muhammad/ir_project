#!/usr/bin/env python3
"""
scripts/00_prepare_samples.py
==============================
Build the ONE immutable sample manifest used by every later stage.

For every attack in the (inherited) grid:
  1. load Exp 01's cached pool (selected_examples.jsonl)
  2. keep successful instances: score_attack - score_control > SKIP_EPSILON
     (imported from src.patching)
  3. seed-42 deterministic sample of up to max_successful_per_attack
  4. resolve attack source positions A (Exp 01 alignment) and query target
     positions Q (Exp 06 spans); validate every invariant (fail loudly)

Outputs (under outputs.base_dir):
  sample_manifest.jsonl        one record per (attack, example)
  00_samples/sample_summary.csv  counts per attack
  00_samples/compute_budget.json estimated forward passes / rows
  00_samples/status.json       includes manifest_sha256 (guards later stages)

Re-running without --force: recomputes the manifest in memory and requires
it to be byte-identical to the existing one (immutability check).
"""

from __future__ import annotations

import argparse
import io
import json
import pathlib
import sys

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
EXP_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(EXP_DIR))

import exp15lib  # noqa: E402,F401  (adds sibling experiment dirs to sys.path)

import pandas as pd  # noqa: E402
from transformers import T5Tokenizer  # noqa: E402

from src.patching import SKIP_EPSILON  # noqa: E402

from exp15lib.config import get_attacks, load_config, output_dir, resolve_cfg_path  # noqa: E402
from exp15lib.heads import load_heads  # noqa: E402
from exp15lib.positions import build_encodings_and_positions  # noqa: E402
from exp15lib.run_utils import MANIFEST_NAME, file_sha256, now, write_status  # noqa: E402
from exp15lib.sampling import load_pool, sample_attack, successful_instances  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Exp 15 stage 00: sample manifest")
    p.add_argument("--config", default=str(EXP_DIR / "configs" / "default.yaml"))
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    out = output_dir(cfg)
    stage = out / "00_samples"
    manifest_path = out / MANIFEST_NAME
    out.mkdir(parents=True, exist_ok=True)

    tok = T5Tokenizer.from_pretrained(cfg["model"]["checkpoint"], use_fast=False)
    attacks = get_attacks(cfg)
    heads = load_heads(cfg)
    exp1_dir = resolve_cfg_path(cfg, cfg["data"]["exp1_attacks_dir"])
    seed = int(cfg["sampling"]["seed"])
    cap = int(cfg["sampling"]["max_successful_per_attack"])
    max_len = int(cfg["model"]["max_length"])
    print(f"[00] {len(attacks)} attacks, {len(heads)} heads, cap={cap}, seed={seed}, SKIP_EPSILON={SKIP_EPSILON}")

    records, summary = [], []
    for spec in attacks:
        pool = load_pool(exp1_dir, spec.attack_name)
        n_succ = len(successful_instances(pool))
        sampled = sample_attack(pool, spec.attack_name, seed, cap)
        if not sampled:
            raise RuntimeError(f"{spec.attack_name}: no successful instances in the Exp 01 pool")
        for i, ex in enumerate(sampled):
            try:
                _, _, info = build_encodings_and_positions(
                    tok, ex["query"], ex["passage"], ex["attacked_passage"], max_len)
            except Exception as exc:
                raise RuntimeError(f"{spec.attack_name} {ex['qid']}/{ex['docid']}: {exc}") from exc
            records.append({
                "attack_name": spec.attack_name, "attack_token": spec.token,
                "attack_position": spec.position, "repetitions": spec.repetitions,
                "sample_index": i, "example_id": f"{ex['qid']}_{ex['docid']}",
                "qid": str(ex["qid"]), "docid": str(ex["docid"]),
                "query": ex["query"], "passage": ex["passage"], "attacked_passage": ex["attacked_passage"],
                "score_control": ex["control_score"], "score_attack": ex["attack_score"],
                "delta": ex["attack_score"] - ex["control_score"],
                "n_attack_positions": len(info["attack_source_positions"]),
                "n_query_tokens": len(info["query_target_positions"]),
                **info,
            })
        sub = [r for r in records if r["attack_name"] == spec.attack_name]
        summary.append({
            "attack_name": spec.attack_name, "attack_token": spec.token,
            "attack_position": spec.position, "repetitions": spec.repetitions,
            "n_pool": len(pool), "n_successful": n_succ, "n_sampled": len(sub),
            "used_all_successful": len(sub) == n_succ,
            "mean_delta": sum(r["delta"] for r in sub) / len(sub),
            "mean_n_query_tokens": sum(r["n_query_tokens"] for r in sub) / len(sub),
            "mean_n_attack_positions": sum(r["n_attack_positions"] for r in sub) / len(sub),
            "mean_n_attack_spans": sum(len(r["attack_spans"]) for r in sub) / len(sub),
            "mean_seq_len": sum(r["seq_len"] for r in sub) / len(sub),
        })
        print(f"  {spec.attack_name:24s} pool={len(pool):3d} successful={n_succ:3d} sampled={len(sub):2d}")

    buf = io.StringIO()
    for r in records:
        buf.write(json.dumps(r, sort_keys=True) + "\n")
    content = buf.getvalue()

    if manifest_path.exists() and not args.force:
        if manifest_path.read_text(encoding="utf-8") != content:
            raise RuntimeError(
                f"Existing {manifest_path} differs from the freshly derived manifest. The manifest is "
                "immutable; rerun with --force ONLY if you intend to invalidate all downstream results."
            )
        print(f"[00] existing manifest verified identical ({len(records)} records)")
    else:
        manifest_path.write_text(content, encoding="utf-8")
        print(f"[00] wrote {manifest_path} ({len(records)} records)")

    stage.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary).to_csv(stage / "sample_summary.csv", index=False)

    # ---- compute budget (rows = hypothetical forward passes) ----------------
    H = len(heads)
    n_layers = len({h.layer for h in heads})
    nq = sum(r["n_query_tokens"] for r in records)
    n_ex = len(records)
    budget = {
        "n_examples": n_ex, "n_heads": H, "n_head_layers": n_layers,
        "total_query_tokens": nq, "mean_query_tokens": nq / n_ex,
        "rows_stage02_all_query": n_ex * H * 2,
        "rows_stage03_single_query": nq * H * 2,
        "rows_stage04_whole_head": n_ex * H * 2,
        "unpatched_capture_passes": n_ex * 2 * 3,
    }
    budget["rows_total"] = sum(v for k, v in budget.items() if k.startswith("rows_")) + budget["unpatched_capture_passes"]
    with open(stage / "compute_budget.json", "w") as fh:
        json.dump(budget, fh, indent=2)
    print("[00] compute budget (encoder+1-step-decoder sequence passes):")
    for k, v in budget.items():
        print(f"      {k:28s} {v:,}" if isinstance(v, int) else f"      {k:28s} {v:.2f}")

    write_status(stage, {
        "status": "success", "finished": now(), "n_records": len(records),
        "n_attacks": len(attacks), "manifest_path": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "seed": seed, "max_successful_per_attack": cap, "skip_epsilon": SKIP_EPSILON,
    })


if __name__ == "__main__":
    main()
