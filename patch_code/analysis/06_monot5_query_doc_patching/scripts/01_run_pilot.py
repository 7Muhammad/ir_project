#!/usr/bin/env python3
"""
scripts/01_run_pilot.py
========================
Experiment 6 timing pilot — REQUIRED before running the full grid
(scripts/02_run_grid.py). Runs 1 attack (relevant_start_5, by default),
n=5 examples, all 12 encoder + 12 decoder layers, all 3 conditions, both
regions (encoder_self_attn, decoder_cross_attn), and Part 1's attention
analysis. Records wall-clock time per (layer, condition) cell, broken down
by region, since encoder self-attention patching (full network re-execution)
and decoder cross-attention patching (encoder-output reuse) have very
different costs.

Prints per-region mean seconds/cell and extrapolates total wall-clock time
for two candidate full-grid sizes:
  (a) Exp-3 tiering:      10 examples x 105 attacks + 100 x canonical attack
  (b) Higher-n alternative: 30 examples x 105 attacks + 100 x canonical attack

Does NOT write output CSVs for downstream aggregation — this is a timing
measurement, not a data-collection run. (Rows are still saved to
outputs_pilot/ for manual inspection/debugging, but scripts/03_aggregate.py
is not intended to run against them.)
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import sys
import time
from typing import Dict, List

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
EXP_DIR = SCRIPT_DIR.parent
EXP1_DIR = EXP_DIR.parent / "01_monot5_layer_patching"
sys.path.insert(0, str(EXP1_DIR))
sys.path.insert(0, str(EXP_DIR))

from src.attack_registry import discover_attacks
from src.model_utils import get_true_false_token_ids, load_monot5, resolve_device

from exp6lib.attention import compute_encoder_attentions, normalized_attention_masses
from exp6lib.engine import run_decoder_cross_attn_example, run_encoder_self_attn_example
from exp6lib.run_utils import build_example_inputs, get_examples_for_attack, load_config, resolve_cfg_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Experiment 6 timing pilot.")
    p.add_argument("--config", default=str(EXP_DIR / "configs" / "pilot.yaml"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(pathlib.Path(args.config))
    outputs_base = resolve_cfg_path(cfg, cfg["outputs"]["base_dir"])
    outputs_base.mkdir(parents=True, exist_ok=True)

    device = resolve_device(cfg["model"]["device"])
    print(f"[pilot] device: {device}")
    model, tokenizer = load_monot5(cfg["model"]["checkpoint"], device)
    true_id, false_id = get_true_false_token_ids(tokenizer)

    canonical_name = cfg["runs"]["canonical"]["attack_name"]
    n_examples = cfg["runs"]["canonical"]["n_examples"]
    single_cfg = {
        "upstream_injected_dir": cfg["attacks"]["upstream_injected_dir"],
        "mode": "include", "include": [canonical_name],
    }
    spec = discover_attacks(single_cfg)[0]
    examples = get_examples_for_attack(spec, n_examples, cfg, model, tokenizer, true_id, false_id, device)
    print(f"[pilot] {len(examples)} examples for {canonical_name}")

    n_enc_layers = model.config.num_layers
    n_dec_layers = model.config.num_decoder_layers
    enc_layers = list(range(n_enc_layers))
    dec_layers = list(range(n_dec_layers))
    n_conditions = len(cfg["conditions"])

    csv_path = outputs_base / "pilot_rows.csv"
    fields = ["region", "layer", "condition", "score_control", "score_attack",
              "score_patched_fwd", "score_patched_rev", "fwd_effect", "rev_effect", "combined_effect"]
    fh = open(csv_path, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()

    enc_total_time = 0.0
    dec_total_time = 0.0
    attn_total_time = 0.0
    n_enc_examples = 0
    n_dec_examples = 0
    n_skipped = 0

    for i, ex in enumerate(examples):
        inputs = build_example_inputs(tokenizer, ex, cfg["model"]["max_length"], device)
        if inputs.status != "ok":
            print(f"  [{i+1}/{len(examples)}] SKIP (align failure): {inputs.reason}")
            n_skipped += 1
            continue

        # Part 1: attention analysis (clean + attack), timed separately.
        t0 = time.time()
        clean_attn = compute_encoder_attentions(model, inputs.clean_enc, device)
        atk_attn = compute_encoder_attentions(model, inputs.attack_enc, device)
        _ = normalized_attention_masses(clean_attn, inputs.query_span, inputs.doc_span_clean)
        _ = normalized_attention_masses(atk_attn, inputs.query_span, inputs.doc_span_control_attack)
        attn_total_time += time.time() - t0

        # Part 2: encoder self-attention, all layers x 3 conditions.
        t0 = time.time()
        enc_rows = run_encoder_self_attn_example(
            model, inputs.control_enc, inputs.attack_enc, enc_layers,
            inputs.query_span, inputs.doc_span_control_attack, true_id, false_id, device,
        )
        enc_elapsed = time.time() - t0
        if enc_rows is not None:
            enc_total_time += enc_elapsed
            n_enc_examples += 1
            writer.writerows(enc_rows)

        # Part 2: decoder cross-attention, all layers x 3 conditions.
        t0 = time.time()
        dec_rows = run_decoder_cross_attn_example(
            model, inputs.control_enc, inputs.attack_enc, dec_layers,
            inputs.query_span, inputs.doc_span_control_attack, true_id, false_id, device,
        )
        dec_elapsed = time.time() - t0
        if dec_rows is not None:
            dec_total_time += dec_elapsed
            n_dec_examples += 1
            writer.writerows(dec_rows)

        print(f"  [{i+1}/{len(examples)}] qid={ex['qid']} docid={ex['docid']} "
              f"enc={enc_elapsed:.2f}s dec={dec_elapsed:.2f}s", flush=True)

    fh.close()

    if n_enc_examples == 0 or n_dec_examples == 0:
        sys.exit("[pilot] ERROR: no examples produced valid rows — cannot extrapolate timing.")

    n_enc_cells = n_enc_layers * n_conditions
    n_dec_cells = n_dec_layers * n_conditions

    sec_per_enc_example = enc_total_time / n_enc_examples
    sec_per_dec_example = dec_total_time / n_dec_examples
    sec_per_attn_example = attn_total_time / len(examples)
    sec_per_enc_cell = sec_per_enc_example / n_enc_cells
    sec_per_dec_cell = sec_per_dec_example / n_dec_cells

    print("\n" + "=" * 70)
    print("  PILOT TIMING SUMMARY")
    print("=" * 70)
    print(f"  Examples processed: {len(examples)}  (skipped: {n_skipped})")
    print(f"  Encoder self-attn : {sec_per_enc_example:.2f} s/example "
          f"({n_enc_cells} cells -> {sec_per_enc_cell*1000:.1f} ms/cell)")
    print(f"  Decoder cross-attn: {sec_per_dec_example:.2f} s/example "
          f"({n_dec_cells} cells -> {sec_per_dec_cell*1000:.1f} ms/cell)")
    print(f"  Attention analysis: {sec_per_attn_example:.2f} s/example (fixed, not per-layer)")

    sec_per_example_total = sec_per_enc_example + sec_per_dec_example + sec_per_attn_example
    print(f"  TOTAL: {sec_per_example_total:.2f} s/example (all regions + attention)")

    def extrapolate(n_per_attack: int, n_attacks: int, n_canonical: int) -> float:
        total_examples = n_per_attack * n_attacks + n_canonical
        return total_examples * sec_per_example_total

    tiering_a = extrapolate(10, 105, 100)
    tiering_b = extrapolate(30, 105, 100)

    print("\n" + "-" * 70)
    print("  EXTRAPOLATED FULL-GRID RUNTIME (single device, same throughput as pilot)")
    print("-" * 70)
    print(f"  (a) Exp-3 tiering   (10x105 + 100 canonical = "
          f"{10*105+100} examples): {tiering_a/3600:.2f} hours")
    print(f"  (b) Higher-n        (30x105 + 100 canonical = "
          f"{30*105+100} examples): {tiering_b/3600:.2f} hours")
    print("-" * 70)
    print("  Do NOT launch scripts/02_run_grid.py until n_examples is confirmed")
    print("  based on these estimates (see configs/default.yaml TODO).")
    print("=" * 70)


if __name__ == "__main__":
    main()
