#!/usr/bin/env python3
"""
scripts/03_layer9_head_attention_crossref.py
================================================
Experiment 11, Step 4 -- per-head document<->query attention mass at
encoder layers 8-11 (layer 9 is the headline target: Experiment 6 found a
1.42x-uniform document->query mass spike there, head-averaged), and its
cross-reference against Step 1's per-head causal-patching results.

Per-head attention matrices from Experiment 6 were NOT saved to disk
(its normalized_attention_masses averages over heads before any CSV write
-- verified by inspecting outputs/canonical/aggregated/attention_summary.csv,
which has no head column). This script recomputes attention fresh, on the
SAME canonical-attack examples Step 1 used (loaded from
outputs/canonical/selected_examples.jsonl for direct comparability), on
both clean and attacked inputs (matching Experiment 6 Part 1's own
clean-vs-attack convention, not the padded control).

Output: outputs/layer9_crossref/results.csv (per-example, per-layer,
per-head mass) and outputs/layer9_crossref/classification.csv (2x2
verdict per (layer, head), reusing Step 1's canonical combined_effect).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
EXP_DIR = SCRIPT_DIR.parent
EXP1_DIR = EXP_DIR.parent / "01_monot5_layer_patching"
EXP6_DIR = EXP_DIR.parent / "06_monot5_query_doc_patching"
sys.path.insert(0, str(EXP1_DIR))   # src.*
sys.path.insert(0, str(EXP6_DIR))   # exp6lib.* (span detection + attention, reused)
sys.path.insert(0, str(EXP_DIR))    # exp11lib.*

import pandas as pd

from src.model_utils import load_monot5, resolve_device

from exp6lib.attention import compute_encoder_attentions
from exp6lib.run_utils import build_example_inputs

from exp11lib.attention_crossref import per_head_masses
from exp11lib.run_utils import load_config, resolve_cfg_path

MASS_BASELINE = 1.0
LAYERS = [8, 9, 10, 11]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Experiment 11 -- Step 4 layer 8-11 per-head attention cross-reference.")
    p.add_argument("--config", default=str(EXP_DIR / "configs" / "default.yaml"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(pathlib.Path(args.config))
    outputs_base = resolve_cfg_path(cfg, cfg["outputs"]["base_dir"])
    out_dir = outputs_base / "layer9_crossref"
    out_dir.mkdir(parents=True, exist_ok=True)

    examples_path = outputs_base / "canonical" / "selected_examples.jsonl"
    if not examples_path.exists():
        sys.exit(f"[03_layer9_head_attention_crossref] {examples_path} not found -- "
                  f"run scripts/01_encoder_head_patch_canonical.py first (same examples reused here).")
    examples = [json.loads(l) for l in open(examples_path, encoding="utf-8")]
    print(f"[03_layer9_head_attention_crossref] {len(examples)} examples (reused from Step 1), "
          f"layers={LAYERS}, per-head attention recomputed fresh (not saved by Experiment 6).")

    device = resolve_device(cfg["model"]["device"])
    model, tokenizer = load_monot5(cfg["model"]["checkpoint"], device)
    max_length = cfg["model"]["max_length"]

    all_rows = []
    n_used = 0
    for i, ex in enumerate(examples):
        inputs = build_example_inputs(tokenizer, ex, max_length, device)
        if inputs.status != "ok":
            continue

        clean_attn = compute_encoder_attentions(model, inputs.clean_enc, device)
        atk_attn = compute_encoder_attentions(model, inputs.attack_enc, device)

        for layer in LAYERS:
            for row in per_head_masses(clean_attn[layer], inputs.query_span, inputs.doc_span_clean):
                all_rows.append({"qid": ex["qid"], "docid": ex["docid"], "encoding": "clean",
                                  "layer": layer, **row})
            for row in per_head_masses(atk_attn[layer], inputs.query_span, inputs.doc_span_control_attack):
                all_rows.append({"qid": ex["qid"], "docid": ex["docid"], "encoding": "attack",
                                  "layer": layer, **row})
        n_used += 1
        if (i + 1) % 20 == 0 or i == len(examples) - 1:
            print(f"    example {i+1}/{len(examples)} (used={n_used})", flush=True)

    df = pd.DataFrame(all_rows)
    df.to_csv(out_dir / "results.csv", index=False)
    print(f"[03_layer9_head_attention_crossref] wrote {len(df)} rows ({n_used} examples) -> {out_dir/'results.csv'}")

    agg = df.groupby(["encoding", "layer", "head"]).agg(
        n=("qid", "size"),
        attention_d_to_q_mean=("attention_d_to_q", "mean"),
        attention_q_to_d_mean=("attention_q_to_d", "mean"),
    ).reset_index()
    agg.to_csv(out_dir / "aggregated.csv", index=False)

    # Cross-reference against Step 1's per-(layer,head) causal effect.
    canonical_csv = outputs_base / "canonical" / "results.csv"
    causal_df = pd.read_csv(canonical_csv)
    causal_agg = causal_df.groupby(["layer", "head_idx"])["combined_effect"].mean().reset_index()

    threshold = cfg["flag_threshold"]
    atk_agg = agg[agg["encoding"] == "attack"].set_index(["layer", "head"])
    classification_rows = []
    for _, row in causal_agg.iterrows():
        layer, head = int(row["layer"]), int(row["head_idx"])
        if (layer, head) not in atk_agg.index:
            continue
        d_to_q = float(atk_agg.loc[(layer, head), "attention_d_to_q_mean"])
        q_to_d = float(atk_agg.loc[(layer, head), "attention_q_to_d_mean"])
        peak_mass = max(d_to_q, q_to_d)
        causal_effect = float(row["combined_effect"])
        mass_is_high = peak_mass > MASS_BASELINE
        causal_is_high = causal_effect > threshold
        if mass_is_high and causal_is_high:
            verdict = "carries both mass and causal effect"
        elif mass_is_high and not causal_is_high:
            verdict = "high mass, low causal effect (descriptive-only, not load-bearing)"
        elif not mass_is_high and causal_is_high:
            verdict = "high causal effect, unremarkable mass (mechanism isn't visible in raw attention)"
        else:
            verdict = "uninvolved"
        classification_rows.append({
            "layer": layer, "head": head,
            "attention_d_to_q": d_to_q, "attention_q_to_d": q_to_d,
            "combined_effect": causal_effect,
            "mass_is_high": mass_is_high, "causal_is_high": causal_is_high,
            "verdict": verdict,
        })
    classification_df = pd.DataFrame(classification_rows).sort_values(
        ["layer", "combined_effect"], ascending=[True, False]
    )
    classification_df.to_csv(out_dir / "classification.csv", index=False)

    print("\n[03_layer9_head_attention_crossref] Layer 9 classification (sorted by combined_effect):")
    print(classification_df[classification_df["layer"] == 9].round(3).to_string(index=False))
    n_both = (classification_df["verdict"] == "carries both mass and causal effect").sum()
    print(f"\n  {n_both} of {len(classification_df)} (layer, head) cells carry both mass and causal effect.")
    print(f"\n  Next: python scripts/04_attack_token_hub_analysis.py --config {args.config}")


if __name__ == "__main__":
    main()
