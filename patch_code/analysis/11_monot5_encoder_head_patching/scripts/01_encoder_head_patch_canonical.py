#!/usr/bin/env python3
"""
scripts/01_encoder_head_patch_canonical.py
=============================================
Experiment 11, Step 1 -- per-head causal patching of monoT5 ENCODER
self-attention, canonical attack (relevant_start_5), n=100, all 144
head-slots (12 layers x 12 heads), whole-sequence patching.

Layers are processed in `heads.layer_priority` order (front-loads 9-11 per
the experiment prompt) but ALL layers still run and are written -- this is
a processing order, not a sub-selection.

Resume-safe (status.json + results.csv), same convention as Experiments 3/6.
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys
import time
import traceback

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
EXP_DIR = SCRIPT_DIR.parent
EXP1_DIR = EXP_DIR.parent / "01_monot5_layer_patching"
sys.path.insert(0, str(EXP1_DIR))   # src.*
sys.path.insert(0, str(EXP_DIR))    # exp11lib.*

from src.attack_registry import discover_attacks
from src.model_utils import (
    build_padded_control_and_attack_encodings_general,
    get_true_false_token_ids,
    load_monot5,
    resolve_device,
)

from exp11lib.engine import run_grid_example
from exp11lib.head_hooks import enumerate_layers
from exp11lib.run_utils import get_examples_for_attack, load_config, resolve_cfg_path

FIELDS = [
    "qid", "docid", "attack_name", "layer", "head_idx",
    "score_control", "score_attack",
    "score_patched_fwd", "score_patched_rev",
    "score_ablated_zero", "score_ablated_mean",
    "fwd_effect", "rev_effect", "combined_effect",
    "score_drop_zero", "score_drop_mean",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Experiment 11 -- Step 1 canonical-attack encoder head patch.")
    p.add_argument("--config", default=str(EXP_DIR / "configs" / "default.yaml"))
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def _write_status(out_dir: pathlib.Path, status: dict) -> None:
    with open(out_dir / "status.json", "w", encoding="utf-8") as fh:
        json.dump(status, fh, indent=2)


def _is_already_successful(out_dir: pathlib.Path) -> bool:
    status_path = out_dir / "status.json"
    if not status_path.exists():
        return False
    try:
        with open(status_path, encoding="utf-8") as fh:
            st = json.load(fh)
    except Exception:
        return False
    return st.get("status") == "success" and (out_dir / "results.csv").exists()


def ordered_layers(model, cfg) -> list:
    heads_cfg = cfg.get("heads", {})
    all_layers = enumerate_layers(model, heads_cfg.get("layers"))
    priority = heads_cfg.get("layer_priority") or []
    ordered = [l for l in priority if l in all_layers]
    ordered += [l for l in all_layers if l not in ordered]
    return ordered


def main() -> None:
    args = parse_args()
    cfg = load_config(pathlib.Path(args.config))
    outputs_base = resolve_cfg_path(cfg, cfg["outputs"]["base_dir"])
    out_dir = outputs_base / "canonical"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[01_encoder_head_patch_canonical] Config:  {args.config}")
    print(f"[01_encoder_head_patch_canonical] Outputs: {out_dir}")

    if not args.force and _is_already_successful(out_dir):
        print("[01_encoder_head_patch_canonical] already successful -- skipping (use --force to re-run).")
        return

    device = resolve_device(cfg["model"]["device"])
    model, tokenizer = load_monot5(cfg["model"]["checkpoint"], device)
    true_id, false_id = get_true_false_token_ids(tokenizer)

    layers = ordered_layers(model, cfg)
    n_heads = model.config.num_heads
    methods = cfg["ablation"]["methods"]
    print(f"[01_encoder_head_patch_canonical] {len(layers)} layers x {n_heads} heads = "
          f"{len(layers) * n_heads} head-slots. Layer order: {layers}")

    canonical_name = cfg["runs"]["canonical"]["attack_name"]
    n_examples = cfg["runs"]["canonical"]["n_examples"]
    max_length = cfg["model"]["max_length"]
    single_cfg = {
        "upstream_injected_dir": cfg["attacks"]["upstream_injected_dir"],
        "mode": "include", "include": [canonical_name],
    }
    spec = discover_attacks(single_cfg)[0]

    examples = get_examples_for_attack(spec, n_examples, cfg, model, tokenizer, true_id, false_id, device)
    with open(out_dir / "selected_examples.jsonl", "w", encoding="utf-8") as fh:
        for ex in examples:
            fh.write(json.dumps(ex, ensure_ascii=False) + "\n")

    status = {
        "attack_name": canonical_name, "status": "failed", "error": None,
        "n_examples_requested": n_examples, "n_examples_used": 0,
        "n_align_failed": 0, "n_skipped_epsilon": 0, "n_rows": 0,
    }
    _write_status(out_dir, status)

    csv_path = out_dir / "results.csv"
    t0 = time.time()
    try:
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
            writer.writeheader()

            for i, ex in enumerate(examples):
                control_enc, attack_enc, align_result = build_padded_control_and_attack_encodings_general(
                    tokenizer=tokenizer, query=ex["query"], passage=ex["passage"],
                    attacked_passage=ex["attacked_passage"], max_length=max_length, device=device,
                )
                if align_result.status != "ok":
                    print(f"    ALIGN FAIL qid={ex['qid']} docid={ex['docid']}: {align_result.reason[:120]}")
                    status["n_align_failed"] += 1
                    continue

                meta = {"qid": ex["qid"], "docid": ex["docid"], "attack_name": canonical_name}
                rows = run_grid_example(
                    model, control_enc, attack_enc, layers, true_id, false_id, device, meta, methods,
                )
                if rows is None:
                    status["n_skipped_epsilon"] += 1
                    continue

                writer.writerows(rows)
                fh.flush()
                status["n_examples_used"] += 1
                status["n_rows"] += len(rows)
                elapsed = time.time() - t0
                print(f"    example {i+1}/{len(examples)} qid={ex['qid']} docid={ex['docid']} "
                      f"rows={len(rows)} ({elapsed/(i+1):.1f}s/ex)", flush=True)

        status["status"] = "success"

    except Exception:
        tb = traceback.format_exc()
        print(f"\nERROR in canonical head patch:\n{tb}")
        status["error"] = tb

    _write_status(out_dir, status)

    if status["status"] != "success":
        sys.exit("[01_encoder_head_patch_canonical] FAILED -- re-run to resume/debug.")
    print(f"\n[01_encoder_head_patch_canonical] Done. examples={status['n_examples_used']} "
          f"rows={status['n_rows']} align_failed={status['n_align_failed']} "
          f"skipped_epsilon={status['n_skipped_epsilon']}")


if __name__ == "__main__":
    main()
