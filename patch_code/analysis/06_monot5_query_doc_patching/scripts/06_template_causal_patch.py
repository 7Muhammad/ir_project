#!/usr/bin/env python3
"""
scripts/06_template_causal_patch.py
=====================================
Experiment 6 extension, Step 1 -- single-position causal patching of the 7
template positions (Query, :, Document, :, Relevant, :, </s>), canonical
attack (relevant_start_5), n=100, all 12 encoder layers, encoder
self-attention only.

Resume-safe (status.json + results.csv), same convention as
scripts/02_run_grid.py: skip if already successful unless --force.

Output: outputs/template_tokens/causal_patch/results.csv (per-example,
per-layer, per-position rows) -- see scripts/09_template_aggregate_and_plot.py
for heatmap/bar-chart/additivity-check derived views.
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
sys.path.insert(0, str(EXP1_DIR))
sys.path.insert(0, str(EXP_DIR))

from src.attack_registry import discover_attacks
from src.model_utils import get_true_false_token_ids, load_monot5, resolve_device

from exp6lib.run_utils import build_example_inputs, get_examples_for_attack, load_config, resolve_cfg_path
from exp6lib.template_engine import run_template_position_causal_patch_example
from exp6lib.template_positions import get_template_positions

FIELDS = [
    "qid", "docid", "attack_name", "region", "layer", "template_position", "n_tokens",
    "score_control", "score_attack", "score_patched_fwd", "score_patched_rev",
    "fwd_effect", "rev_effect", "combined_effect",
    "query_span_start", "query_span_end", "doc_span_start", "doc_span_end",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Experiment 6 extension -- Step 1 template-position causal patch.")
    p.add_argument("--config", default=str(EXP_DIR / "configs" / "template_tokens.yaml"))
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


def main() -> None:
    args = parse_args()
    cfg = load_config(pathlib.Path(args.config))
    outputs_base = resolve_cfg_path(cfg, cfg["outputs"]["base_dir"])
    out_dir = outputs_base / "causal_patch"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[06_template_causal_patch] Config:  {args.config}")
    print(f"[06_template_causal_patch] Outputs: {out_dir}")

    if not args.force and _is_already_successful(out_dir):
        print("[06_template_causal_patch] already successful -- skipping (use --force to re-run).")
        return

    device = resolve_device(cfg["model"]["device"])
    model, tokenizer = load_monot5(cfg["model"]["checkpoint"], device)
    true_id, false_id = get_true_false_token_ids(tokenizer)

    canonical_name = cfg["runs"]["canonical"]["attack_name"]
    n_examples = cfg["runs"]["canonical"]["n_examples"]
    max_length = cfg["model"]["max_length"]
    single_cfg = {
        "upstream_injected_dir": cfg["attacks"]["upstream_injected_dir"],
        "mode": "include", "include": [canonical_name],
    }
    spec = discover_attacks(single_cfg)[0]

    encoder_layers = list(range(model.config.num_layers))
    print(f"[06_template_causal_patch] attack={canonical_name} n_examples={n_examples} "
          f"layers={len(encoder_layers)} positions=7 (encoder_self_attn only)")

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
                inputs = build_example_inputs(tokenizer, ex, max_length, device)
                if inputs.status != "ok":
                    print(f"    ALIGN FAIL qid={ex['qid']} docid={ex['docid']}: {inputs.reason[:120]}")
                    status["n_align_failed"] += 1
                    continue

                attack_ids = inputs.attack_enc["input_ids"][0].tolist()
                template_positions = get_template_positions(
                    tokenizer, attack_ids, inputs.query_span, inputs.doc_span_control_attack,
                )

                rows = run_template_position_causal_patch_example(
                    model, inputs.control_enc, inputs.attack_enc, encoder_layers,
                    template_positions, true_id, false_id, device,
                )
                if not rows:
                    status["n_skipped_epsilon"] += 1
                    continue

                for row in rows:
                    row["qid"] = ex["qid"]
                    row["docid"] = ex["docid"]
                    row["attack_name"] = canonical_name
                    row["query_span_start"], row["query_span_end"] = inputs.query_span
                    row["doc_span_start"], row["doc_span_end"] = inputs.doc_span_control_attack

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
        print(f"\nERROR in template causal patch:\n{tb}")
        status["error"] = tb

    _write_status(out_dir, status)

    if status["status"] != "success":
        sys.exit("[06_template_causal_patch] FAILED -- re-run to resume/debug.")
    print(f"\n[06_template_causal_patch] Done. examples={status['n_examples_used']} "
          f"rows={status['n_rows']} align_failed={status['n_align_failed']} "
          f"skipped_epsilon={status['n_skipped_epsilon']}")
    print(f"  Next: python scripts/07_template_attention_mass.py --config {args.config}")


if __name__ == "__main__":
    main()
