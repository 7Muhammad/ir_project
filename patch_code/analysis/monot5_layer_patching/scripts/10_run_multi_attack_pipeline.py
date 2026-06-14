#!/usr/bin/env python3
"""
scripts/10_run_multi_attack_pipeline.py
========================================
Multi-attack orchestrator: runs the full activation-patching pipeline
(Stages 00–04) for each attack discovered via the attack registry.

Key design decisions
---------------------
  - The model is loaded ONCE and reused across all attacks.
    This avoids 7× repeated model loads (~2 min each on CPU, seconds on GPU).

  - Each attack writes to its own isolated output directory:
        outputs/attacks/{attack_name}/
            pairs/
            scores/
            activations/
            patching/
            plots/
            status.json

  - If an attack fails (any stage), the exception is caught, logged to
    status.json, and the runner continues to the next attack.

  - Activation caches are deleted after Stage 03 if
    attacks.cleanup_activations_after_patching is true (default: true).
    This saves ~90 MB × max_cached_examples per attack on disk.

  - alignment failures are counted and reported in status.json.
    Examples where alignment fails are skipped (not silently dropped):
    the failure count is always printed.

Status file schema (outputs/attacks/{attack_name}/status.json)
---------------------------------------------------------------
{
  "attack_name": "relevant_start_5",
  "token": "relevant",
  "position": "start",
  "repetitions": 5,
  "run_name": "bm25_19",
  "status": "success" | "failed" | "skipped",
  "error": null | "<traceback string>",
  "n_pairs_loaded": 0,
  "n_align_ok": 0,
  "n_align_failed": 0,
  "n_scored": 0,
  "n_selected": 0
}

Usage
-----
  python scripts/10_run_multi_attack_pipeline.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import shutil
import sys
import traceback
from typing import Dict, List, Optional

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import yaml

from src.activation_hooks import cache_all_activations_from_enc
from src.attack_registry import AttackSpec, discover_attacks
from src.data_utils import load_attacked_tsv, save_pairs
from src.model_utils import (
    build_padded_control_and_attack_encodings_general,
    build_monot5_input,
    get_true_false_token_ids,
    load_monot5,
    resolve_device,
    score_from_encoding,
)
from src.patching import aggregate_results, run_layer_patching_for_example
from src.plotting import (
    plot_attack_delta_histogram,
    plot_combined_heatmap,
    plot_patching_heatmap,
)
from src.scoring import score_batch, select_attacked_examples


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Multi-attack orchestrator: run the full pipeline for each attack."
    )
    p.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "default.yaml"),
        help="Path to the YAML config file.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Status helpers
# ---------------------------------------------------------------------------

_EMPTY_STATUS = {
    "attack_name": "",
    "token": "",
    "position": "",
    "repetitions": 0,
    "run_name": "",
    "status": "failed",
    "error": None,
    "n_pairs_loaded": 0,
    "n_align_ok": 0,
    "n_align_failed": 0,
    "n_scored": 0,
    "n_selected": 0,
}


def _write_status(attack_dir: pathlib.Path, status: dict) -> None:
    status_path = attack_dir / "status.json"
    with open(status_path, "w", encoding="utf-8") as fh:
        json.dump(status, fh, indent=2)


def _is_already_successful(attack_dir: pathlib.Path) -> bool:
    """
    Return True if this attack already completed successfully in a previous run.

    We check two things:
      1. status.json exists and contains ``"status": "success"``.
      2. The key output files exist (layer_patching_results.csv and at least one
         plot), so a partial / corrupted run is not mistakenly skipped.
    """
    status_path = attack_dir / "status.json"
    if not status_path.exists():
        return False
    try:
        with open(status_path, encoding="utf-8") as fh:
            st = json.load(fh)
    except Exception:
        return False

    if st.get("status") != "success":
        return False

    # Require key outputs to be present as a sanity check.
    required = [
        attack_dir / "patching" / "layer_patching_results.csv",
        attack_dir / "scores" / "all_scores.csv",
    ]
    if not all(p.exists() for p in required):
        return False

    # At least one plot must exist.
    plots_dir = attack_dir / "plots"
    if not plots_dir.is_dir() or not any(plots_dir.iterdir()):
        return False

    return True


# ---------------------------------------------------------------------------
# Per-attack stage logic
# ---------------------------------------------------------------------------

def _run_stage_00(
    spec: AttackSpec,
    attack_dir: pathlib.Path,
    cfg: dict,
) -> int:
    """
    Stage 00: load the pre-built attacked TSV and write pairs/pairs.jsonl.

    Returns the number of pairs loaded.
    """
    data_cfg    = cfg["data"]
    runtime_cfg = cfg["runtime"]

    pairs_out = attack_dir / "pairs" / "pairs.jsonl"
    pairs_out.parent.mkdir(parents=True, exist_ok=True)

    if not spec.path.exists():
        raise FileNotFoundError(
            f"Attacked TSV not found: {spec.path}"
        )

    pairs = load_attacked_tsv(
        path=spec.path,
        max_pairs=data_cfg["max_pairs"],
        max_docs_per_query=data_cfg["max_docs_per_query"],
        seed=runtime_cfg["seed"],
    )
    save_pairs(pairs, pairs_out)
    print(f"  [stage00] {len(pairs)} pairs → {pairs_out}")
    return len(pairs)


def _run_stage_01(
    attack_dir: pathlib.Path,
    cfg: dict,
    model,
    tokenizer,
    true_id: int,
    false_id: int,
    device: torch.device,
) -> tuple[int, int, int]:
    """
    Stage 01: score pairs and select examples where attack > control.

    Uses build_padded_control_and_attack_encodings_general so all attack
    positions (start, end, random) are handled uniformly.

    Returns (n_scored, n_selected, n_align_failed).
    """
    model_cfg     = cfg["model"]
    selection_cfg = cfg["selection"]
    runtime_cfg   = cfg["runtime"]
    attacks_cfg   = cfg.get("attacks", {})
    max_cached    = attacks_cfg.get("max_cached_examples") or runtime_cfg["max_cached_examples"]

    pairs_path = attack_dir / "pairs" / "pairs.jsonl"
    scores_dir = attack_dir / "scores"
    scores_dir.mkdir(parents=True, exist_ok=True)

    # Load pairs
    pairs = []
    with open(pairs_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))
    print(f"  [stage01] Loaded {len(pairs)} pairs.")

    max_length = model_cfg["max_length"]

    # Score all three variants (original, control, attack) for each pair.
    # Original: text-based, batched.
    # Control + Attack: token-level, per-example using general alignment.
    scored: List[Dict] = []
    n_align_failed = 0

    # --- Batch-score original clean inputs ---
    original_texts = [build_monot5_input(p["query"], p["passage"]) for p in pairs]
    batch_size = runtime_cfg["batch_size_scoring"]
    original_scores = score_batch(
        model=model,
        tokenizer=tokenizer,
        input_texts=original_texts,
        true_id=true_id,
        false_id=false_id,
        max_length=max_length,
        device=device,
        batch_size=batch_size,
    )
    print(f"  [stage01] Scored {len(original_scores)} original inputs.")

    # --- Per-example: build general padded-control + attack, score both ---
    for i, pair in enumerate(pairs):
        if i % 50 == 0:
            print(f"  [stage01] Scoring control/attack: {i}/{len(pairs)} ...", flush=True)

        query   = pair["query"]
        passage = pair["passage"]
        attacked_passage = pair.get("attacked_passage", "")

        if not attacked_passage:
            print(f"    WARNING: no attacked_passage for qid={pair['qid']} docid={pair['docid']}, skipping.")
            n_align_failed += 1
            continue

        control_enc, attack_enc, align_result = build_padded_control_and_attack_encodings_general(
            tokenizer=tokenizer,
            query=query,
            passage=passage,
            attacked_passage=attacked_passage,
            max_length=max_length,
            device=device,
        )

        if align_result.status != "ok":
            print(
                f"    ALIGN FAIL qid={pair['qid']} docid={pair['docid']}: "
                f"{align_result.reason[:120]}"
            )
            n_align_failed += 1
            continue

        control_score = score_from_encoding(model, control_enc, true_id, false_id, device)
        attack_score  = score_from_encoding(model, attack_enc,  true_id, false_id, device)

        delta_ca  = attack_score  - control_score
        delta_oa  = attack_score  - original_scores[i]
        delta_co  = control_score - original_scores[i]

        rec = {
            **pair,
            "original_score":              original_scores[i],
            "control_score":               control_score,
            "attack_score":                attack_score,
            "control_delta_vs_original":   delta_co,
            "attack_delta_vs_original":    delta_oa,
            "attack_delta_vs_control":     delta_ca,
        }
        scored.append(rec)

    print(
        f"  [stage01] Scored {len(scored)} pairs successfully. "
        f"Alignment failures: {n_align_failed}."
    )

    if n_align_failed > 0:
        pct = 100 * n_align_failed / max(1, len(pairs))
        print(
            f"  [stage01] WARNING: {n_align_failed}/{len(pairs)} pairs "
            f"({pct:.1f}%) failed alignment and were skipped."
        )

    # Save all scores
    all_scores_path = scores_dir / "all_scores.csv"
    fieldnames = [
        "qid", "docid", "rank", "bm25_score",
        "original_score", "control_score", "attack_score",
        "control_delta_vs_original", "attack_delta_vs_control", "attack_delta_vs_original",
    ]
    with open(all_scores_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(scored)

    # Select examples
    selected = select_attacked_examples(
        scored_pairs=scored,
        min_attack_delta=selection_cfg["min_attack_delta"],
        max_selected=selection_cfg["max_selected_examples"],
        seed=runtime_cfg["seed"],
    )

    selected_path = scores_dir / "selected_examples.jsonl"
    with open(selected_path, "w", encoding="utf-8") as fh:
        for ex in selected:
            fh.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"  [stage01] {len(selected)} selected → {selected_path}")
    return len(scored), len(selected), n_align_failed


def _run_stage_02(
    attack_dir: pathlib.Path,
    cfg: dict,
    model,
    tokenizer,
    device: torch.device,
) -> int:
    """
    Stage 02: cache control and attack activations for selected examples.

    Returns the number of examples cached.
    """
    model_cfg   = cfg["model"]
    runtime_cfg = cfg["runtime"]
    attacks_cfg = cfg.get("attacks", {})
    max_cached  = attacks_cfg.get("max_cached_examples") or runtime_cfg["max_cached_examples"]
    max_length  = model_cfg["max_length"]

    selected_path = attack_dir / "scores" / "selected_examples.jsonl"
    act_dir       = attack_dir / "activations"
    act_dir.mkdir(parents=True, exist_ok=True)

    examples = []
    with open(selected_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                examples.append(json.loads(line))

    if len(examples) > max_cached:
        examples = examples[:max_cached]
        print(f"  [stage02] Truncated to {max_cached} (max_cached_examples).")

    for i, ex in enumerate(examples):
        qid     = ex["qid"]
        docid   = ex["docid"]
        query   = ex["query"]
        passage = ex["passage"]
        attacked_passage = ex["attacked_passage"]

        print(f"  [stage02] [{i+1}/{len(examples)}] qid={qid} docid={docid}", flush=True)

        control_enc, attack_enc, align_result = build_padded_control_and_attack_encodings_general(
            tokenizer=tokenizer,
            query=query,
            passage=passage,
            attacked_passage=attacked_passage,
            max_length=max_length,
            device=device,
        )

        if align_result.status != "ok":
            print(f"    ALIGN FAIL: {align_result.reason[:120]} — skipping.")
            continue

        control_cache = cache_all_activations_from_enc(model=model, enc=control_enc, device=device)
        torch.save(
            {"activations": control_cache,
             "encoding": {k: v.cpu() for k, v in control_enc.items()}},
            act_dir / f"{qid}_{docid}_control.pt",
        )

        attack_cache = cache_all_activations_from_enc(model=model, enc=attack_enc, device=device)
        torch.save(
            {"activations": attack_cache,
             "encoding": {k: v.cpu() for k, v in attack_enc.items()}},
            act_dir / f"{qid}_{docid}_attack.pt",
        )

    print(f"  [stage02] Done. Activations in {act_dir}")
    return len(examples)


def _run_stage_03(
    attack_dir: pathlib.Path,
    cfg: dict,
    model,
    true_id: int,
    false_id: int,
    device: torch.device,
) -> None:
    """Stage 03: run layer-level activation patching for all cached examples."""
    runtime_cfg = cfg["runtime"]
    attacks_cfg = cfg.get("attacks", {})
    max_cached  = attacks_cfg.get("max_cached_examples") or runtime_cfg["max_cached_examples"]

    selected_path = attack_dir / "scores" / "selected_examples.jsonl"
    act_dir       = attack_dir / "activations"
    patch_dir     = attack_dir / "patching"
    patch_dir.mkdir(parents=True, exist_ok=True)

    examples: List[Dict] = []
    with open(selected_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    if len(examples) > max_cached:
        examples = examples[:max_cached]

    all_detailed: List[Dict] = []
    skipped = 0

    for i, ex in enumerate(examples):
        qid   = ex["qid"]
        docid = ex["docid"]
        print(
            f"  [stage03] [{i+1}/{len(examples)}] qid={qid} docid={docid} "
            f"delta={ex['attack_delta_vs_control']:.4f}",
            flush=True,
        )

        ctrl_pt   = act_dir / f"{qid}_{docid}_control.pt"
        attack_pt = act_dir / f"{qid}_{docid}_attack.pt"

        if not ctrl_pt.exists() or not attack_pt.exists():
            print(f"    WARNING: activations not found — skipping.")
            skipped += 1
            continue

        ctrl_data   = torch.load(ctrl_pt,   map_location="cpu", weights_only=True)
        attack_data = torch.load(attack_pt, map_location="cpu", weights_only=True)

        results = run_layer_patching_for_example(
            example=ex,
            model=model,
            control_enc=ctrl_data["encoding"],
            attack_enc=attack_data["encoding"],
            control_cache=ctrl_data["activations"],
            attack_cache=attack_data["activations"],
            true_id=true_id,
            false_id=false_id,
            device=device,
        )
        if results is None:
            skipped += 1
            continue
        all_detailed.extend(results)

    print(f"  [stage03] Processed {len(examples) - skipped} examples ({skipped} skipped).")

    if not all_detailed:
        print("  [stage03] WARNING: no patching results — skipping CSV output.")
        return

    # Save detailed results
    detailed_path = patch_dir / "layer_patching_results_detailed.csv"
    detailed_fields = [
        "qid", "docid", "layer", "component", "direction",
        "control_score", "attack_score", "patched_score", "effect",
    ]
    with open(detailed_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=detailed_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_detailed)

    # Aggregate
    agg = aggregate_results(all_detailed)
    agg_path = patch_dir / "layer_patching_results.csv"
    agg_fields = ["layer", "component", "direction", "mean_effect", "std_effect", "n"]
    with open(agg_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=agg_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(agg)

    print(f"  [stage03] Saved results → {patch_dir}")


def _run_stage_04(
    attack_dir: pathlib.Path,
    cfg: dict,
) -> None:
    """Stage 04: generate per-attack plots."""
    import pandas as pd

    selection_cfg = cfg["selection"]
    plots_dir     = attack_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    selected_path    = attack_dir / "scores" / "selected_examples.jsonl"
    agg_results_path = attack_dir / "patching" / "layer_patching_results.csv"

    # Histogram
    if selected_path.exists():
        selected = []
        with open(selected_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    selected.append(json.loads(line))
        plot_attack_delta_histogram(
            scored_pairs=selected,
            output_path=plots_dir / "attack_delta_hist.png",
            min_attack_delta=selection_cfg["min_attack_delta"],
        )

    # Heatmaps
    if not agg_results_path.exists():
        print(f"  [stage04] No patching results found — skipping heatmaps.")
        return

    agg_df = pd.read_csv(agg_results_path)
    plot_patching_heatmap(agg_df, direction="forward",
                          output_path=plots_dir / "forward_layer_component_heatmap.png")
    plot_patching_heatmap(agg_df, direction="reverse",
                          output_path=plots_dir / "reverse_layer_component_heatmap.png")
    plot_combined_heatmap(agg_df, output_path=plots_dir / "combined_layer_component_heatmap.png")
    print(f"  [stage04] Plots → {plots_dir}")


# ---------------------------------------------------------------------------
# Per-attack orchestration
# ---------------------------------------------------------------------------

def run_single_attack(
    spec: AttackSpec,
    cfg: dict,
    outputs_base: pathlib.Path,
    model,
    tokenizer,
    true_id: int,
    false_id: int,
    device: torch.device,
) -> dict:
    """
    Run all 5 pipeline stages for one attack and return the status dict.
    """
    attacks_cfg = cfg.get("attacks", {})
    cleanup = attacks_cfg.get("cleanup_activations_after_patching", True)

    attack_dir = outputs_base / "attacks" / spec.attack_name
    attack_dir.mkdir(parents=True, exist_ok=True)

    status = {**_EMPTY_STATUS, **{
        "attack_name": spec.attack_name,
        "token":       spec.token,
        "position":    spec.position,
        "repetitions": spec.repetitions,
        "run_name":    spec.run_name,
    }}

    _write_status(attack_dir, status)  # write "in-progress" status immediately

    try:
        print(f"\n{'='*60}")
        print(f"  Attack: {spec.attack_name}")
        print(f"  File:   {spec.path.name}")
        print(f"  Dir:    {attack_dir}")
        print(f"{'='*60}")

        n_pairs = _run_stage_00(spec, attack_dir, cfg)
        status["n_pairs_loaded"] = n_pairs

        if n_pairs == 0:
            status["status"] = "skipped"
            status["error"]  = "No pairs loaded from attacked TSV."
            _write_status(attack_dir, status)
            return status

        n_scored, n_selected, n_align_failed = _run_stage_01(
            attack_dir, cfg, model, tokenizer, true_id, false_id, device
        )
        status["n_scored"]       = n_scored
        status["n_selected"]     = n_selected
        status["n_align_failed"] = n_align_failed

        if n_selected == 0:
            status["status"] = "skipped"
            status["error"]  = "No examples selected after scoring."
            _write_status(attack_dir, status)
            return status

        _run_stage_02(attack_dir, cfg, model, tokenizer, device)
        _run_stage_03(attack_dir, cfg, model, true_id, false_id, device)

        if cleanup:
            act_dir = attack_dir / "activations"
            if act_dir.exists():
                shutil.rmtree(act_dir)
                print(f"  [cleanup] Removed {act_dir}")

        _run_stage_04(attack_dir, cfg)

        status["status"] = "success"
        status["error"]  = None

    except Exception:
        tb = traceback.format_exc()
        print(f"\nERROR in attack '{spec.attack_name}':\n{tb}")
        status["status"] = "failed"
        status["error"]  = tb

    _write_status(attack_dir, status)
    return status


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    cfg_path = pathlib.Path(args.config).resolve()
    print(f"[10_run_multi_attack_pipeline] Config: {cfg_path}")

    with open(cfg_path) as fh:
        cfg = yaml.safe_load(fh)

    attacks_cfg = cfg.get("attacks")
    if not attacks_cfg:
        sys.exit(
            "ERROR: 'attacks' block not found in config.\n"
            "Add an 'attacks:' section — see configs/default.yaml for the template."
        )

    outputs_base = PROJECT_ROOT / cfg["outputs"]["base_dir"]

    # --- Discover attacks ---
    print("[10_run_multi_attack_pipeline] Discovering attacks ...")
    specs = discover_attacks(attacks_cfg)
    print(f"[10_run_multi_attack_pipeline] Found {len(specs)} attacks:")

    # --- Resume: classify each spec as pending or already done ---
    pending_specs: List[AttackSpec] = []
    resumed_statuses: List[dict] = []
    for s in specs:
        attack_dir = outputs_base / "attacks" / s.attack_name
        if _is_already_successful(attack_dir):
            status_path = attack_dir / "status.json"
            with open(status_path, encoding="utf-8") as fh:
                cached = json.load(fh)
            print(f"  [RESUME] {s.attack_name:35s}  already done — skipping.")
            resumed_statuses.append(cached)
        else:
            print(f"  [TODO  ] {s.attack_name:35s}  {s.path.name}")
            pending_specs.append(s)

    print(
        f"\n[10_run_multi_attack_pipeline] "
        f"{len(pending_specs)} attacks pending, {len(resumed_statuses)} already complete."
    )

    all_statuses: List[dict] = list(resumed_statuses)

    if not pending_specs:
        print("[10_run_multi_attack_pipeline] Nothing to do — all attacks already complete.")
    else:
        # --- Load model ONCE (only if there is work to do) ---
        device = resolve_device(cfg["model"]["device"])
        print(f"\n[10_run_multi_attack_pipeline] Loading model: {cfg['model']['checkpoint']}")
        model, tokenizer = load_monot5(cfg["model"]["checkpoint"], device)
        true_id, false_id = get_true_false_token_ids(tokenizer)
        print(f"[10_run_multi_attack_pipeline] Model ready. true_id={true_id}, false_id={false_id}")

        # --- Run each pending attack ---
        for spec in pending_specs:
            status = run_single_attack(
                spec=spec,
                cfg=cfg,
                outputs_base=outputs_base,
                model=model,
                tokenizer=tokenizer,
                true_id=true_id,
                false_id=false_id,
                device=device,
            )
            all_statuses.append(status)

    # --- Summary ---
    print(f"\n{'='*60}")
    print("  MULTI-ATTACK PIPELINE SUMMARY")
    print(f"{'='*60}")
    for s in all_statuses:
        icon = "✓" if s["status"] == "success" else ("~" if s["status"] == "skipped" else "✗")
        print(
            f"  {icon} {s['attack_name']:35s} "
            f"status={s['status']:8s}  "
            f"scored={s['n_scored']:4d}  "
            f"selected={s['n_selected']:4d}  "
            f"align_failed={s['n_align_failed']:3d}"
        )

    n_success  = sum(1 for s in all_statuses if s["status"] == "success")
    n_failed   = sum(1 for s in all_statuses if s["status"] == "failed")
    n_skipped  = sum(1 for s in all_statuses if s["status"] == "skipped")
    n_resumed  = len(resumed_statuses)
    print(f"\n  Success: {n_success}  Failed: {n_failed}  Skipped: {n_skipped}  Resumed (already done): {n_resumed}")
    print(f"\n  Outputs: {outputs_base / 'attacks'}")
    print(f"\n  Next step: python scripts/11_compare_attacks.py --config {cfg_path}")


if __name__ == "__main__":
    main()
