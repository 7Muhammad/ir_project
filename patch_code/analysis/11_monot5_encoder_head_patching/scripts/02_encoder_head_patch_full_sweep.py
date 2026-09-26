#!/usr/bin/env python3
"""
scripts/02_encoder_head_patch_full_sweep.py
===============================================
Experiment 11, Step 2 -- full 105-attack sweep of the Step 1 per-head causal
patch, n=10/attack (matching Experiment 6's template-position sweep
convention), all 144 head-slots, not pre-filtered by the canonical result.

Then Step 3 -- additivity check, prediction stated before computing:
Experiment 6 found the template-position sum was SUB-additive (0.44x of the
whole-span effect). We predict the same pattern here (heads interact rather
than contributing independently) and report whether that prediction holds.
Sub-additivity, if found, is NOT an anomaly -- it would match the project's
existing pattern and is reported as such either way.

Resume-safe per attack (status.json + results.csv), same convention as
scripts/01 and Experiments 3/6.
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys
import time
import traceback
from typing import List

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
EXP_DIR = SCRIPT_DIR.parent
EXP1_DIR = EXP_DIR.parent / "01_monot5_layer_patching"
sys.path.insert(0, str(EXP1_DIR))
sys.path.insert(0, str(EXP_DIR))

import pandas as pd

from src.attack_registry import AttackSpec, discover_attacks
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
LATE_LAYERS = [9, 10, 11]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Experiment 11 -- Step 2 full sweep + Step 3 additivity.")
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


def process_attack(spec: AttackSpec, cfg: dict, out_dir: pathlib.Path, model, tokenizer,
                    true_id, false_id, device, layers: List[int], methods: List[str]) -> dict:
    n_examples = cfg["runs"]["sweep"]["n_examples"]
    max_length = cfg["model"]["max_length"]

    out_dir.mkdir(parents=True, exist_ok=True)
    status = {
        "attack_name": spec.attack_name, "status": "failed", "error": None,
        "n_examples_requested": n_examples, "n_examples_used": 0,
        "n_align_failed": 0, "n_skipped_epsilon": 0, "n_rows": 0,
    }
    _write_status(out_dir, status)

    try:
        examples = get_examples_for_attack(spec, n_examples, cfg, model, tokenizer, true_id, false_id, device)
        with open(out_dir / "selected_examples.jsonl", "w", encoding="utf-8") as fh:
            for ex in examples:
                fh.write(json.dumps(ex, ensure_ascii=False) + "\n")

        csv_path = out_dir / "results.csv"
        t0 = time.time()
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
            writer.writeheader()

            for i, ex in enumerate(examples):
                control_enc, attack_enc, align_result = build_padded_control_and_attack_encodings_general(
                    tokenizer=tokenizer, query=ex["query"], passage=ex["passage"],
                    attacked_passage=ex["attacked_passage"], max_length=max_length, device=device,
                )
                if align_result.status != "ok":
                    status["n_align_failed"] += 1
                    continue

                meta = {"qid": ex["qid"], "docid": ex["docid"], "attack_name": spec.attack_name}
                rows = run_grid_example(
                    model, control_enc, attack_enc, layers, true_id, false_id, device, meta, methods,
                )
                if not rows:
                    status["n_skipped_epsilon"] += 1
                    continue

                writer.writerows(rows)
                fh.flush()
                status["n_examples_used"] += 1
                status["n_rows"] += len(rows)

            elapsed = time.time() - t0
            print(f"    [{spec.attack_name}] {status['n_examples_used']}/{len(examples)} examples, "
                  f"{status['n_rows']} rows ({elapsed/max(1,len(examples)):.1f}s/ex)", flush=True)

        status["status"] = "success"

    except Exception:
        tb = traceback.format_exc()
        print(f"\nERROR in {spec.attack_name}:\n{tb}")
        status["error"] = tb

    _write_status(out_dir, status)
    return status


def additivity_check(cfg: dict, outputs_base: pathlib.Path, sweep_df: pd.DataFrame) -> str:
    """
    Step 3. Flag heads whose full-sweep peak (layers 9-11) mean combined_effect
    exceeds cfg['flag_threshold']; sum those heads' CANONICAL (n=100, Step 1)
    peak effects; compare against the existing whole-layer 'both'/encoder
    effect from Experiment 1/6 at the same layers.

    Prediction stated before computing (per the experiment prompt): expect
    sub-additivity, matching Experiment 6's 0.44x finding for template
    positions. Report whether that prediction holds either way.
    """
    threshold = cfg["flag_threshold"]
    late_sweep = sweep_df[sweep_df["layer"].isin(LATE_LAYERS)]
    sweep_peak = (late_sweep.groupby(["head_idx", "layer"])["combined_effect"].mean()
                            .groupby("head_idx").max())
    flagged_heads = sweep_peak[sweep_peak > threshold].index.tolist()

    canonical_csv = outputs_base / "canonical" / "results.csv"
    if not canonical_csv.exists():
        return (f"[additivity] SKIPPED -- {canonical_csv} not found. Run "
                f"scripts/01_encoder_head_patch_canonical.py first.")
    canonical_df = pd.read_csv(canonical_csv)
    late_canon = canonical_df[canonical_df["layer"].isin(LATE_LAYERS)]
    canon_peak = (late_canon.groupby(["head_idx", "layer"])["combined_effect"].mean()
                            .groupby("head_idx").max())
    sum_flagged_peaks = float(sum(canon_peak.get(h, 0.0) for h in flagged_heads))

    # Whole-layer comparison: Experiment 6's whole-span 'both' condition,
    # encoder_self_attn, same layers (already available, read-only).
    exp6_summary_path = (outputs_base.parent.parent / "06_monot5_query_doc_patching"
                          / "outputs" / "canonical" / "aggregated" / "layer_condition_summary.csv")
    lines = [
        f"Prediction (stated before computing): sub-additive, matching Experiment 6's "
        f"template-position finding (0.44x of whole-span effect).",
        f"Flagged heads (full-sweep peak combined_effect > {threshold}, layers 9-11): "
        f"{len(flagged_heads)} of {sweep_peak.shape[0]} -> {sorted(flagged_heads)}",
        f"Sum of flagged heads' individual peak combined_effect (canonical, n=100): {sum_flagged_peaks:.4f}",
    ]
    if exp6_summary_path.exists():
        exp6_summary = pd.read_csv(exp6_summary_path)
        both_enc = exp6_summary[(exp6_summary["region"] == "encoder_self_attn") & (exp6_summary["condition"] == "both")]
        both_enc_late = both_enc[both_enc["layer"].isin(LATE_LAYERS)]
        whole_span_peak = float(both_enc_late.set_index("layer")["combined_effect_mean"].max())
        ratio = sum_flagged_peaks / whole_span_peak if whole_span_peak else float("nan")
        verdict = "sub-additive (prediction HOLDS)" if ratio < 0.9 else (
            "roughly additive (prediction DOES NOT hold)" if 0.9 <= ratio <= 1.3 else
            "super-additive (prediction DOES NOT hold, unexpected)")
        lines.append(f"Whole-span 'both' (query+document) peak combined_effect at layers 9-11 "
                     f"(Experiment 6, encoder_self_attn): {whole_span_peak:.4f}")
        lines.append(f"Ratio: {ratio:.3f}x -- {verdict}")
        lines.append(
            "Consequence for Phase 4: " + (
                "sub-additivity means individual head patches understate the whole-span effect, "
                "so a real combinatorial SEARCH over head combinations is needed (a greedy "
                "top-k-by-individual-effect selection would likely underperform) -- Phase 4 needs "
                "a search algorithm, not a shortcut." if ratio < 0.9 else
                "effects compose close to additively, so a search algorithm may be unnecessary -- "
                "top-k heads by individual effect likely suffice for Phase 4, which can be scoped down."
            )
        )
    else:
        lines.append(f"[additivity] Whole-span comparison SKIPPED -- {exp6_summary_path} not found.")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    cfg = load_config(pathlib.Path(args.config))
    outputs_base = resolve_cfg_path(cfg, cfg["outputs"]["base_dir"])
    sweep_dir = outputs_base / "sweep"
    print(f"[02_encoder_head_patch_full_sweep] Config:  {args.config}")
    print(f"[02_encoder_head_patch_full_sweep] Outputs: {sweep_dir}")

    device = resolve_device(cfg["model"]["device"])
    model, tokenizer = load_monot5(cfg["model"]["checkpoint"], device)
    true_id, false_id = get_true_false_token_ids(tokenizer)

    layers = enumerate_layers(model, cfg.get("heads", {}).get("layers"))
    methods = cfg["ablation"]["methods"]
    specs = discover_attacks(cfg["attacks"])
    print(f"[02_encoder_head_patch_full_sweep] {len(specs)} attack(s), "
          f"n_examples={cfg['runs']['sweep']['n_examples']}/attack, "
          f"{len(layers)} layers x 12 heads")

    all_statuses = []
    for spec in specs:
        out_dir = sweep_dir / "attacks" / spec.attack_name
        if not args.force and _is_already_successful(out_dir):
            print(f"  [RESUME] {spec.attack_name} already done -- skipping.")
            with open(out_dir / "status.json", encoding="utf-8") as fh:
                all_statuses.append(json.load(fh))
            continue
        print(f"  [START ] {spec.attack_name}")
        all_statuses.append(process_attack(
            spec, cfg, out_dir, model, tokenizer, true_id, false_id, device, layers, methods,
        ))

    print(f"\n{'='*60}\n  SUMMARY\n{'='*60}")
    for s in all_statuses:
        icon = "OK" if s["status"] == "success" else "FAIL"
        print(f"  [{icon:4s}] {s['attack_name']:30s} examples={s['n_examples_used']:4d} "
              f"rows={s['n_rows']:7d} align_failed={s['n_align_failed']}")
    n_failed = sum(1 for s in all_statuses if s["status"] != "success")
    if n_failed:
        sys.exit(f"\n{n_failed} attack(s) failed -- re-run to resume.")

    print(f"\n{'='*60}\n  STEP 3: ADDITIVITY CHECK\n{'='*60}")
    sweep_csvs = sorted(sweep_dir.glob("attacks/*/results.csv"))
    sweep_df = pd.concat([pd.read_csv(p) for p in sweep_csvs], ignore_index=True)
    report = additivity_check(cfg, outputs_base, sweep_df)
    print(report)
    (outputs_base / "additivity_check.txt").write_text(report + "\n", encoding="utf-8")
    print(f"\n  Next: python scripts/03_layer9_head_attention_crossref.py --config {args.config}")


if __name__ == "__main__":
    main()
