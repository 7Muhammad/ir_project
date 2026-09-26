#!/usr/bin/env python3
"""
scripts/01_extract_diffs.py
=============================
Experiment 10, stage 1: for each selected attack, for each reused example,
compute the per-position encoder-output diff vector (attack - control) at
every configured layer, tag each position, and write the pooled
(all-examples, all-positions) diff matrix to disk.

Resume-safe: an attack whose status.json says "success" and whose
diffs.npz exists is skipped unless --force is given.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
import traceback
from typing import Dict, List

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
EXP_DIR = SCRIPT_DIR.parent
EXP1_DIR = EXP_DIR.parent / "01_monot5_layer_patching"
EXP6_DIR = EXP_DIR.parent / "06_monot5_query_doc_patching"
sys.path.insert(0, str(EXP1_DIR))   # src.*
sys.path.insert(0, str(EXP6_DIR))   # exp6lib.* (span detection, reused)
sys.path.insert(0, str(EXP_DIR))    # exp10lib.*

import numpy as np

from src.model_utils import get_true_false_token_ids, load_monot5, resolve_device

from exp10lib.engine import compute_example_diffs
from exp10lib.run_utils import build_example_inputs, load_reused_examples, load_config, resolve_cfg_path, select_attacks


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Experiment 10 stage 1: extract per-position diffs.")
    p.add_argument("--config", default=str(EXP_DIR / "configs" / "default.yaml"))
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def _write_status(attack_dir: pathlib.Path, status: dict) -> None:
    with open(attack_dir / "status.json", "w", encoding="utf-8") as fh:
        json.dump(status, fh, indent=2)


def _is_already_successful(attack_dir: pathlib.Path) -> bool:
    status_path = attack_dir / "status.json"
    if not status_path.exists():
        return False
    try:
        with open(status_path, encoding="utf-8") as fh:
            st = json.load(fh)
    except Exception:
        return False
    return st.get("status") == "success" and (attack_dir / "diffs.npz").exists()


def process_attack(attack_name: str, cfg: dict, out_dir: pathlib.Path, model, tokenizer, device, layers: List[int]) -> dict:
    n_examples = cfg["selection"]["n_examples"]
    max_length = cfg["model"]["max_length"]
    reuse_dir = resolve_cfg_path(cfg, cfg["selection"]["reuse_dir"])

    out_dir.mkdir(parents=True, exist_ok=True)
    status = {
        "attack_name": attack_name, "status": "failed", "error": None,
        "n_examples_requested": n_examples, "n_examples_used": 0,
        "n_align_failed": 0, "n_span_failed": 0, "n_positions": 0,
    }
    _write_status(out_dir, status)

    try:
        examples = load_reused_examples(attack_name, n_examples, reuse_dir)

        per_layer_rows: Dict[int, List[np.ndarray]] = {L: [] for L in layers}
        tags_all: List[str] = []
        example_idx_all: List[int] = []

        t0 = time.time()
        for i, ex in enumerate(examples):
            inputs = build_example_inputs(tokenizer, ex, max_length, device)
            if inputs.status != "ok":
                status["n_align_failed"] += 1
                continue

            result_status, reason, diffs, tags = compute_example_diffs(
                model, tokenizer, ex["query"], ex["passage"], inputs, layers,
            )
            if result_status != "ok":
                status["n_span_failed"] += 1
                continue

            n_pos = len(tags)
            for L in layers:
                per_layer_rows[L].append(diffs[L].astype(np.float32))
            tags_all.extend(tags)
            example_idx_all.extend([i] * n_pos)

            status["n_examples_used"] += 1
            elapsed = time.time() - t0
            print(f"    [{attack_name}] example {i+1}/{len(examples)} "
                  f"qid={ex['qid']} docid={ex['docid']} seq_len={inputs.seq_len} "
                  f"({elapsed/(i+1):.2f}s/ex)", flush=True)

        if status["n_examples_used"] == 0:
            raise RuntimeError(
                f"No examples produced valid diffs for '{attack_name}' "
                f"(align_failed={status['n_align_failed']}, span_failed={status['n_span_failed']})."
            )

        save_kwargs = {f"diff_layer{L}": np.concatenate(per_layer_rows[L], axis=0) for L in layers}
        save_kwargs["tags"] = np.array(tags_all, dtype=object)
        save_kwargs["example_idx"] = np.array(example_idx_all, dtype=np.int32)
        np.savez_compressed(out_dir / "diffs.npz", **save_kwargs)

        status["n_positions"] = len(tags_all)
        status["status"] = "success"

    except Exception:
        tb = traceback.format_exc()
        print(f"\nERROR in {attack_name}:\n{tb}")
        status["error"] = tb

    _write_status(out_dir, status)
    return status


def main() -> None:
    args = parse_args()
    cfg = load_config(pathlib.Path(args.config))
    outputs_base = resolve_cfg_path(cfg, cfg["outputs"]["base_dir"])
    layers = cfg["layers"]
    print(f"[01_extract_diffs] Config:  {args.config}")
    print(f"[01_extract_diffs] Layers:  {layers}")
    print(f"[01_extract_diffs] Outputs: {outputs_base}")

    device = resolve_device(cfg["model"]["device"])
    model, tokenizer = load_monot5(cfg["model"]["checkpoint"], device)
    get_true_false_token_ids(tokenizer)  # sanity check only; scoring not needed here

    attack_names = select_attacks(cfg)

    diffs_dir = outputs_base / "diffs"
    all_statuses = []
    for attack_name in attack_names:
        out_dir = diffs_dir / attack_name
        if not args.force and _is_already_successful(out_dir):
            print(f"  [RESUME] {attack_name} already done — skipping.")
            with open(out_dir / "status.json", encoding="utf-8") as fh:
                all_statuses.append(json.load(fh))
            continue
        print(f"  [START ] {attack_name}")
        all_statuses.append(process_attack(attack_name, cfg, out_dir, model, tokenizer, device, layers))

    print(f"\n{'='*60}\n  SUMMARY\n{'='*60}")
    for s in all_statuses:
        icon = "✓" if s["status"] == "success" else "✗"
        print(f"  {icon} {s['attack_name']:22s} status={s['status']:8s} "
              f"examples={s['n_examples_used']:4d} positions={s['n_positions']:6d} "
              f"align_failed={s['n_align_failed']} span_failed={s['n_span_failed']}")
    n_failed = sum(1 for s in all_statuses if s["status"] != "success")
    if n_failed:
        sys.exit(f"\n{n_failed} attack(s) failed — re-run to resume.")
    print(f"\n  Next: python scripts/02_pca_summary.py --config {args.config}")


if __name__ == "__main__":
    main()
