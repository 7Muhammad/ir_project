#!/usr/bin/env python3
"""
scripts/01_run_pilot.py
=========================
Experiment 7 timing pilot — REQUIRED before running the full grid
(scripts/02_run_grid.py). Runs 1 attack (relevant_start_5, by default),
n=5 examples, all 12 decoder layers (whole-block) and Experiment-3-flagged
heads (per-head), for clean/control/attack inputs. Records wall-clock time
per (layer, example) and (head, example) cell separately, since whole-block
and per-head contribution capture have different costs (whole-block reuses
one cached forward pass for all 12 layers; each flagged head needs its own
forward pass with a pre-hook on that layer's `.o` projection).

Extrapolates total wall-clock time for two candidate grid sizes and prints
both — does not write output for downstream aggregation (see
scripts/02_run_grid.py for that).
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import sys
import time

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
EXP_DIR = SCRIPT_DIR.parent
EXP1_DIR = EXP_DIR.parent / "01_monot5_layer_patching"
sys.path.insert(0, str(EXP1_DIR))
sys.path.insert(0, str(EXP_DIR))

from src.attack_registry import discover_attacks
from src.model_utils import get_true_false_token_ids, load_monot5, resolve_device

from exp7lib.engine import run_example
from exp7lib.run_utils import build_example_inputs, get_examples_for_attack, load_config, load_flagged_heads, resolve_cfg_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Experiment 7 timing pilot.")
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

    flagged_heads = load_flagged_heads(cfg)

    canonical_name = cfg["runs"]["canonical"]["attack_name"]
    n_examples = cfg["runs"]["canonical"]["n_examples"]
    single_cfg = {
        "upstream_injected_dir": cfg["attacks"]["upstream_injected_dir"],
        "mode": "include", "include": [canonical_name],
    }
    spec = discover_attacks(single_cfg)[0]
    examples = get_examples_for_attack(spec, n_examples, cfg, model, tokenizer, true_id, false_id, device)
    print(f"[pilot] {len(examples)} examples for {canonical_name}, {len(flagged_heads)} flagged heads")

    n_layers = model.config.num_decoder_layers
    k_values = cfg["k_values"]
    composition_k = cfg["composition_k"]

    csv_path = outputs_base / "pilot_rows.csv"
    fh = open(csv_path, "w", newline="", encoding="utf-8")
    writer = None

    whole_block_total_time = 0.0
    per_head_total_time = 0.0
    n_processed = 0
    n_skipped = 0

    for i, ex in enumerate(examples):
        inputs = build_example_inputs(tokenizer, ex, spec.token, cfg["model"]["max_length"], device)
        if inputs.status != "ok":
            print(f"  [{i+1}/{len(examples)}] SKIP: {inputs.reason}")
            n_skipped += 1
            continue

        # Time whole-block-only first (flagged_heads=[]) to isolate its cost...
        t0 = time.time()
        whole_rows = run_example(
            model, tokenizer, inputs, ex["query"], ex["passage"], spec.token,
            flagged_heads=[], k_values=k_values, composition_k=composition_k, device=device,
        )
        whole_elapsed = time.time() - t0
        whole_block_total_time += whole_elapsed

        # ...then time per-head ADDITIONALLY (flagged heads only) to isolate that cost.
        t0 = time.time()
        head_rows = run_example(
            model, tokenizer, inputs, ex["query"], ex["passage"], spec.token,
            flagged_heads=flagged_heads, k_values=k_values, composition_k=composition_k, device=device,
        ) if flagged_heads else []
        # head_rows above re-includes whole-block rows too (run_example always does both);
        # per-head-only time = total time for (whole+head) minus whole-only time already measured.
        combined_elapsed = time.time() - t0
        per_head_elapsed = max(0.0, combined_elapsed - whole_elapsed)
        per_head_total_time += per_head_elapsed

        rows = head_rows if flagged_heads else whole_rows
        for row in rows:
            row["qid"], row["docid"], row["attack_name"] = ex["qid"], ex["docid"], spec.attack_name
        if writer is None:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()), extrasaction="ignore")
            writer.writeheader()
        writer.writerows(rows)
        fh.flush()

        n_processed += 1
        print(f"  [{i+1}/{len(examples)}] qid={ex['qid']} docid={ex['docid']} "
              f"whole_block={whole_elapsed:.2f}s per_head_extra={per_head_elapsed:.2f}s", flush=True)

    fh.close()

    if n_processed == 0:
        sys.exit("[pilot] ERROR: no examples processed — cannot extrapolate timing.")

    n_run_types = 3  # clean, control, attack
    n_whole_cells = n_layers * n_run_types
    n_head_cells = len(flagged_heads) * n_run_types

    sec_per_example_whole = whole_block_total_time / n_processed
    sec_per_example_head = per_head_total_time / n_processed
    sec_per_whole_cell = sec_per_example_whole / n_whole_cells
    sec_per_head_cell = (sec_per_example_head / n_head_cells) if n_head_cells else 0.0

    print("\n" + "=" * 70)
    print("  PILOT TIMING SUMMARY")
    print("=" * 70)
    print(f"  Examples processed: {n_processed}  (skipped: {n_skipped})")
    print(f"  Flagged heads: {len(flagged_heads)}")
    print(f"  Whole-block: {sec_per_example_whole:.2f} s/example "
          f"({n_whole_cells} cells -> {sec_per_whole_cell*1000:.1f} ms/cell)")
    print(f"  Per-head   : {sec_per_example_head:.2f} s/example "
          f"({n_head_cells} cells -> {sec_per_head_cell*1000:.1f} ms/cell)")

    sec_per_example_total = sec_per_example_whole + sec_per_example_head
    print(f"  TOTAL: {sec_per_example_total:.2f} s/example (whole-block + per-head, all 3 run types)")

    def extrapolate(n_per_attack: int, n_attacks: int, n_canonical: int) -> float:
        total_examples = n_per_attack * n_attacks + n_canonical
        return total_examples * sec_per_example_total

    tiering_a = extrapolate(10, 105, 100)
    tiering_b = extrapolate(30, 105, 100)

    print("\n" + "-" * 70)
    print("  EXTRAPOLATED FULL-GRID RUNTIME (single device, same throughput as pilot)")
    print("-" * 70)
    print(f"  (a) Same tiering as Exp 3/6 (10x105 + 100 canonical = "
          f"{10*105+100} examples): {tiering_a/3600:.2f} hours")
    print(f"  (b) Alternative, 30/attack (30x105 + 100 canonical = "
          f"{30*105+100} examples): {tiering_b/3600:.2f} hours")
    print("-" * 70)
    print("  Do NOT launch scripts/02_run_grid.py until n_examples is confirmed.")
    print("=" * 70)


if __name__ == "__main__":
    main()
