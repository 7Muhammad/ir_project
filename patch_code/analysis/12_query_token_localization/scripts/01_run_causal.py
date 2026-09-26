#!/usr/bin/env python3
"""
scripts/01_run_causal.py
==========================
Experiment 12, Part B (causal): whole-encoder-self-attention-output patching
at each query-word span and each structural control group, for every
successful example (delta > 1e-4) of every attack in scope, across all
requested encoder layers.

Checkpointing / resume
-------------------------
Per attack: outputs/causal/attacks/<attack_name>/{status.json,results.csv}.
On resume (no --force), examples whose (qid, docid) already have rows in
results.csv are skipped -- an attack interrupted partway through only
re-does its last, unwritten example, never previously-written ones. An
attack whose status.json says "success" is skipped entirely.

Debugging a single attack/example: set configs/default.yaml's
`attacks.include: [...]` to one name and `sampling.causal.n_examples_per_attack: 1`,
or pass --config pointing at configs/smoke.yaml.
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys
import time
import traceback
from typing import Dict, List, Set, Tuple

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
EXP_DIR = SCRIPT_DIR.parent
EXP1_DIR = EXP_DIR.parent / "01_monot5_layer_patching"
EXP6_DIR = EXP_DIR.parent / "06_monot5_query_doc_patching"
sys.path.insert(0, str(EXP1_DIR))
sys.path.insert(0, str(EXP6_DIR))
sys.path.insert(0, str(EXP_DIR))

from src.attack_registry import AttackSpec, discover_attacks
from src.model_utils import get_true_false_token_ids, load_monot5, resolve_device

from exp12lib.causal_engine import run_causal_patch_units_example
from exp12lib.run_utils import build_exp12_example_inputs, get_example_pool, load_config, resolve_cfg_path

FIELDS = [
    "attack_name", "qid", "docid", "layer",
    "intervention_type", "intervention_name",
    "query_word_index", "query_word_text", "query_word_token_start", "query_word_token_end", "num_subtokens",
    "content_or_stopword", "matched_or_unmatched", "word_group",
    "score_control", "score_attack", "delta",
    "score_forward_patched", "score_reverse_patched",
    "forward_effect", "reverse_effect", "combined_effect",
]

WORD_UNIT_PREFIX = "qw"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Experiment 12 -- causal query-word/structural patching.")
    p.add_argument("--config", default=str(EXP_DIR / "configs" / "default.yaml"))
    p.add_argument("--force", action="store_true", help="Ignore existing checkpoints; recompute everything.")
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


def build_units(inputs, structural_enabled: bool) -> Tuple[Dict[str, List[int]], Dict[str, dict]]:
    """Returns (units dict for the engine, metadata dict keyed by unit_name)."""
    units: Dict[str, List[int]] = {}
    meta: Dict[str, dict] = {}
    for w in inputs.words:
        name = f"{WORD_UNIT_PREFIX}{w.index}"
        units[name] = w.token_indices
        meta[name] = {
            "intervention_type": "query_word",
            "intervention_name": w.text,
            "query_word_index": w.index,
            "query_word_text": w.text,
            "query_word_token_start": w.token_start,
            "query_word_token_end": w.token_end,
            "num_subtokens": w.num_subtokens,
            "content_or_stopword": w.content_or_stopword,
            "matched_or_unmatched": w.matched_or_unmatched,
            "word_group": w.word_group,
        }
    if structural_enabled:
        for name, indices in inputs.structural_units.items():
            units[name] = indices
            meta[name] = {
                "intervention_type": "structural_group",
                "intervention_name": name,
                "query_word_index": None, "query_word_text": None,
                "query_word_token_start": None, "query_word_token_end": None, "num_subtokens": None,
                "content_or_stopword": None, "matched_or_unmatched": None, "word_group": None,
            }
    return units, meta


def process_attack(spec: AttackSpec, cfg: dict, out_dir: pathlib.Path, model, tokenizer,
                    true_id, false_id, device, layers: List[int], causal_n: int,
                    structural_enabled: bool, force: bool) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "results.csv"
    status = {
        "attack_name": spec.attack_name, "status": "failed", "error": None,
        "n_examples_requested": causal_n, "n_examples_used": 0,
        "n_align_failed": 0, "n_skipped_epsilon": 0, "n_rows": 0,
    }
    _write_status(out_dir, status)

    already_done = set() if force else _completed_examples(csv_path)
    write_mode = "w" if (force or not csv_path.exists()) else "a"

    try:
        examples = get_example_pool(spec, cfg, model, tokenizer, true_id, false_id, device, pool_size=causal_n)
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

                units, meta = build_units(inputs, structural_enabled)
                result = run_causal_patch_units_example(
                    model, inputs.control_enc, inputs.attack_enc, layers, units, true_id, false_id, device,
                )
                if result is None:
                    status["n_skipped_epsilon"] += 1
                    continue
                rows, control_score, attack_score = result

                out_rows = []
                for row in rows:
                    m = meta[row["unit_name"]]
                    out_rows.append({
                        "attack_name": spec.attack_name, "qid": qid, "docid": docid, "layer": row["layer"],
                        **m,
                        "score_control": control_score, "score_attack": attack_score,
                        "delta": attack_score - control_score,
                        "score_forward_patched": row["score_patched_fwd"],
                        "score_reverse_patched": row["score_patched_rev"],
                        "forward_effect": row["fwd_effect"], "reverse_effect": row["rev_effect"],
                        "combined_effect": row["combined_effect"],
                    })
                writer.writerows(out_rows)
                fh.flush()
                status["n_examples_used"] += 1
                status["n_rows"] += len(out_rows)
                elapsed = time.time() - t0
                print(f"    [{spec.attack_name}] example {i+1}/{len(examples)} qid={qid} docid={docid} "
                      f"rows={len(out_rows)} ({elapsed/(i+1):.1f}s/ex)", flush=True)

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
    out_dir_root = outputs_base / "causal" / "attacks"

    if not cfg["enable"]["causal"]:
        print("[01_run_causal] enable.causal is false in config -- nothing to do.")
        return

    print(f"[01_run_causal] Config:  {args.config}")
    print(f"[01_run_causal] Outputs: {out_dir_root}")

    device = resolve_device(cfg["model"]["device"])
    model, tokenizer = load_monot5(cfg["model"]["checkpoint"], device)
    true_id, false_id = get_true_false_token_ids(tokenizer)

    layers = cfg.get("layers") or list(range(model.config.num_layers))
    causal_n = cfg["sampling"]["causal"]["n_examples_per_attack"]
    structural_enabled = cfg["enable"]["structural_controls"]

    specs = discover_attacks(cfg["attacks"])
    print(f"[01_run_causal] {len(specs)} attack(s), n={causal_n}/attack, "
          f"{len(layers)} layer(s), structural_controls={structural_enabled}")

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
            layers, causal_n, structural_enabled, args.force,
        ))

    print(f"\n{'='*60}\n  SUMMARY\n{'='*60}")
    for s in all_statuses:
        icon = "✓" if s["status"] == "success" else "✗"
        print(f"  {icon} {s['attack_name']:30s} status={s['status']:8s} "
              f"examples={s['n_examples_used']:4d} rows={s['n_rows']:7d} align_failed={s['n_align_failed']}")
    n_failed = sum(1 for s in all_statuses if s["status"] != "success")
    if n_failed:
        sys.exit(f"\n{n_failed} attack(s) failed -- re-run to resume.")
    print(f"\n  Next: python scripts/02_run_attention.py --config {args.config}")


if __name__ == "__main__":
    main()
