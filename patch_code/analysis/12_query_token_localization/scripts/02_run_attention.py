#!/usr/bin/env python3
"""
scripts/02_run_attention.py
=============================
Experiment 12, Part A (descriptive): attack-token <-> query-word attention,
both directions, control vs. attack, at layer-averaged and important-head
granularity, for every successful example (delta > 1e-4) of every attack in
scope.

Cheap by construction: exactly 2 encoder forward passes per example
(control, attack, both with output_attentions=True) -- no patched re-runs.
Uses a LARGER independent sample size than the causal run (see
configs/default.yaml sampling.attention.n_examples_per_attack).

Checkpointing / resume: same (attack, example) granularity as
scripts/01_run_causal.py -- see that script's module docstring.
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys
import time
import traceback
from typing import List, Set, Tuple

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
EXP_DIR = SCRIPT_DIR.parent
EXP1_DIR = EXP_DIR.parent / "01_monot5_layer_patching"
EXP6_DIR = EXP_DIR.parent / "06_monot5_query_doc_patching"
sys.path.insert(0, str(EXP1_DIR))
sys.path.insert(0, str(EXP6_DIR))
sys.path.insert(0, str(EXP_DIR))

from src.attack_registry import AttackSpec, discover_attacks
from src.model_utils import get_true_false_token_ids, load_monot5, resolve_device
from src.patching import SKIP_EPSILON

from exp12lib.attention_engine import run_attention_example
from exp12lib.important_heads import load_important_heads
from exp12lib.run_utils import build_exp12_example_inputs, get_example_pool, load_config, resolve_cfg_path

FIELDS = [
    "attack_name", "qid", "docid", "layer", "granularity", "head",
    "query_word_index", "query_word_text", "content_or_stopword", "matched_or_unmatched", "word_group",
    "num_attack_tokens",
    "attack_to_query_control", "attack_to_query_attack", "attack_to_query_delta",
    "attack_to_query_control_mean_per_token", "attack_to_query_attack_mean_per_token",
    "query_to_attack_control", "query_to_attack_attack", "query_to_attack_delta",
    "query_to_attack_control_mean_per_token", "query_to_attack_attack_mean_per_token",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Experiment 12 -- attack<->query attention analysis.")
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
    return st.get("status") == "success" and (attack_dir / "results.csv").exists()


def _completed_examples(csv_path: pathlib.Path) -> Set[Tuple[str, str]]:
    if not csv_path.exists():
        return set()
    done: Set[Tuple[str, str]] = set()
    with open(csv_path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            done.add((row["qid"], row["docid"]))
    return done


def process_attack(spec: AttackSpec, cfg: dict, out_dir: pathlib.Path, model, tokenizer,
                    true_id, false_id, device, layers: List[int], attention_n: int,
                    important_heads, force: bool) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "results.csv"
    status = {
        "attack_name": spec.attack_name, "status": "failed", "error": None,
        "n_examples_requested": attention_n, "n_examples_used": 0,
        "n_align_failed": 0, "n_skipped_epsilon": 0, "n_rows": 0,
    }
    _write_status(out_dir, status)

    already_done = set() if force else _completed_examples(csv_path)
    write_mode = "w" if (force or not csv_path.exists()) else "a"

    try:
        examples = get_example_pool(spec, cfg, model, tokenizer, true_id, false_id, device, pool_size=attention_n)
        with open(out_dir / "selected_examples.jsonl", "w", encoding="utf-8") as fh:
            for ex in examples:
                fh.write(json.dumps(ex, ensure_ascii=False) + "\n")

        t0 = time.time()
        with open(csv_path, write_mode, newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
            if write_mode == "w":
                writer.writeheader()

            for i, ex in enumerate(examples):
                qid, docid = str(ex["qid"]), str(ex["docid"])
                if (qid, docid) in already_done:
                    continue

                inputs = build_exp12_example_inputs(tokenizer, ex, cfg["model"]["max_length"], device)
                if inputs.status != "ok":
                    print(f"    ALIGN FAIL qid={qid} docid={docid}: {inputs.reason[:120]}")
                    status["n_align_failed"] += 1
                    continue

                delta = ex["attack_delta_vs_control"]
                if abs(delta) < SKIP_EPSILON:
                    status["n_skipped_epsilon"] += 1
                    continue

                rows = run_attention_example(
                    model, inputs.control_enc, inputs.attack_enc, inputs.words,
                    inputs.attack_span_indices, layers, important_heads, device,
                )
                out_rows = [{"attack_name": spec.attack_name, "qid": qid, "docid": docid, **row} for row in rows]
                writer.writerows(out_rows)
                fh.flush()
                status["n_examples_used"] += 1
                status["n_rows"] += len(out_rows)
                elapsed = time.time() - t0
                print(f"    [{spec.attack_name}] example {i+1}/{len(examples)} qid={qid} docid={docid} "
                      f"rows={len(out_rows)} ({elapsed/(i+1):.2f}s/ex)", flush=True)

        status["status"] = "success"

    except Exception:
        tb = traceback.format_exc()
        print(f"\nERROR in {spec.attack_name}:\n{tb}")
        status["error"] = tb

    _write_status(out_dir, status)
    return status


def main() -> None:
    args = parse_args()
    cfg = load_config(pathlib.Path(args.config))
    outputs_base = resolve_cfg_path(cfg, cfg["outputs"]["base_dir"])
    out_dir_root = outputs_base / "attention" / "attacks"

    if not cfg["enable"]["attention"]:
        print("[02_run_attention] enable.attention is false in config -- nothing to do.")
        return

    print(f"[02_run_attention] Config:  {args.config}")
    print(f"[02_run_attention] Outputs: {out_dir_root}")

    device = resolve_device(cfg["model"]["device"])
    model, tokenizer = load_monot5(cfg["model"]["checkpoint"], device)
    true_id, false_id = get_true_false_token_ids(tokenizer)

    layers = cfg.get("layers") or list(range(model.config.num_layers))
    attention_n = cfg["sampling"]["attention"]["n_examples_per_attack"]

    important_heads = load_important_heads(
        resolve_cfg_path(cfg, cfg["important_heads"]["source_csv"]),
        top_n=cfg["important_heads"].get("top_n"),
    )
    print(f"[02_run_attention] Important heads (Exp. 11, read-only): "
          f"{[h.label for h in important_heads]}")

    specs = discover_attacks(cfg["attacks"])
    print(f"[02_run_attention] {len(specs)} attack(s), n={attention_n}/attack, {len(layers)} layer(s)")

    all_statuses: List[dict] = []
    for spec in specs:
        out_dir = out_dir_root / spec.attack_name
        if not args.force and _is_already_successful(out_dir):
            print(f"  [RESUME] {spec.attack_name} already done -- skipping.")
            with open(out_dir / "status.json", encoding="utf-8") as fh:
                all_statuses.append(json.load(fh))
            continue
        print(f"  [START ] {spec.attack_name}")
        all_statuses.append(process_attack(
            spec, cfg, out_dir, model, tokenizer, true_id, false_id, device,
            layers, attention_n, important_heads, args.force,
        ))

    print(f"\n{'='*60}\n  SUMMARY\n{'='*60}")
    for s in all_statuses:
        icon = "✓" if s["status"] == "success" else "✗"
        print(f"  {icon} {s['attack_name']:30s} status={s['status']:8s} "
              f"examples={s['n_examples_used']:4d} rows={s['n_rows']:7d} align_failed={s['n_align_failed']}")
    n_failed = sum(1 for s in all_statuses if s["status"] != "success")
    if n_failed:
        sys.exit(f"\n{n_failed} attack(s) failed -- re-run to resume.")
    print(f"\n  Next: python scripts/03_aggregate.py --config {args.config}")


if __name__ == "__main__":
    main()
