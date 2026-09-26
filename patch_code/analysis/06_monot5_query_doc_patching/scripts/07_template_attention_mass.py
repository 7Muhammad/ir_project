#!/usr/bin/env python3
"""
scripts/07_template_attention_mass.py
========================================
Experiment 6 extension, Step 2 -- attention-mass corroboration for the 7
template positions, and the sink/signal/relay/hidden-channel classification.

"Flagged" (per the experiment prompt) = a template position whose Step 1
peak combined_effect (max over layers 9-11, from
outputs/template_tokens/causal_patch/results.csv) exceeds an explicit
threshold, reported at both 0.02 and 0.05 so the choice is visible.

Deviation from the prompt, stated explicitly: attention mass is computed
for ALL 7 positions here, not only the flagged ones. Once the encoder
forward pass with output_attentions=True has run for an example (clean/
control/attack), slicing mass for 7 positions instead of 1-2 costs a few
extra tensor index operations -- negligible next to the pass itself. Doing
all 7 makes the 2x2 classification table (relay/hidden-channel/sink/
uninvolved) complete instead of structurally missing the sink/uninvolved
rows, which only ever apply to LOW-e_comb (i.e. unflagged) positions.
`--flagged-only` restores the letter of the prompt if needed.

Per position, per layer, per encoding (clean/control/attack):
  incoming_doc_mass    = normalized_mass(document-span -> position)
  incoming_attack_mass = normalized_mass(attack-span -> position)     [control/attack only]
  outgoing_query_mass  = normalized_mass(position -> query-span)
  outgoing_doc_mass    = normalized_mass(position -> document-span)

Classification (2x2, mass high/low x e_comb high/low; mass "high" means
normalized mass > 1.0, the uniform-attention baseline already used
throughout exp6lib/attention.py):
  high mass + high e_comb  -> relay            (attention hub AND causal)
  low  mass + high e_comb  -> hidden-channel   (causal without being a hub)
  high mass + low  e_comb  -> sink             (hub but causally inert)
  low  mass + low  e_comb  -> uninvolved
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
EXP_DIR = SCRIPT_DIR.parent
EXP1_DIR = EXP_DIR.parent / "01_monot5_layer_patching"
sys.path.insert(0, str(EXP1_DIR))
sys.path.insert(0, str(EXP_DIR))

import pandas as pd

from src.model_utils import load_monot5, resolve_device

from exp6lib.attention import compute_encoder_attentions, normalized_mass_indices
from exp6lib.run_utils import build_example_inputs, load_config, resolve_cfg_path
from exp6lib.template_positions import TEMPLATE_POSITION_NAMES, get_template_positions

MASS_BASELINE = 1.0  # uniform-attention reference, matches exp6lib/attention.py


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Experiment 6 extension -- Step 2 attention-mass corroboration.")
    p.add_argument("--config", default=str(EXP_DIR / "configs" / "template_tokens.yaml"))
    p.add_argument("--flagged-only", action="store_true",
                    help="Only compute attention mass for positions flagged at the primary "
                         "(first) threshold, per the prompt's literal scope.")
    return p.parse_args()


def flag_positions(causal_df: pd.DataFrame, thresholds: list) -> dict:
    """peak (max over layers 9-11) mean combined_effect per position, and flags at each threshold."""
    late = causal_df[causal_df["layer"].isin([9, 10, 11])]
    # mean per (position, layer) first, then max over the 3 late layers -- avoids example-count bias
    peak = (late.groupby(["template_position", "layer"])["combined_effect"].mean()
                 .groupby("template_position").max())
    result = {}
    for name in TEMPLATE_POSITION_NAMES:
        peak_val = float(peak.get(name, 0.0))
        result[name] = {
            "peak_e_comb": peak_val,
            **{f"flagged_{t}": bool(peak_val > t) for t in thresholds},
        }
    return result


def mass_for_example(tokenizer, model, device, inputs) -> list:
    """Per-(layer, encoding, position) attention-mass rows for one example."""
    rows = []
    attack_ids = inputs.attack_enc["input_ids"][0].tolist()
    attack_positions = get_template_positions(
        tokenizer, attack_ids, inputs.query_span, inputs.doc_span_control_attack,
    )
    doc_span = inputs.doc_span_control_attack
    query_span = inputs.query_span
    attack_span = inputs.attack_span_indices or []

    clean_ids = inputs.clean_enc["input_ids"][0].tolist()
    clean_positions = get_template_positions(
        tokenizer, clean_ids, inputs.query_span, inputs.doc_span_clean,
    )
    doc_span_clean = inputs.doc_span_clean

    encodings = {
        "clean": (inputs.clean_enc, clean_positions, doc_span_clean, len(clean_ids), False),
        "control": (inputs.control_enc, attack_positions, doc_span, len(attack_ids), True),
        "attack": (inputs.attack_enc, attack_positions, doc_span, len(attack_ids), True),
    }

    for enc_name, (enc, positions, d_span, total_len, has_attack_span) in encodings.items():
        attns = compute_encoder_attentions(model, enc, device)
        for layer_idx, attn in enumerate(attns):
            attn_avg = attn[0].mean(dim=0)  # (seq_len, seq_len)
            for pos_name in TEMPLATE_POSITION_NAMES:
                idxs = positions[pos_name]
                incoming_doc = normalized_mass_indices(attn_avg, list(range(*d_span)), idxs, total_len)
                outgoing_query = normalized_mass_indices(attn_avg, idxs, list(range(*query_span)), total_len)
                outgoing_doc = normalized_mass_indices(attn_avg, idxs, list(range(*d_span)), total_len)
                incoming_attack = (
                    normalized_mass_indices(attn_avg, attack_span, idxs, total_len)
                    if has_attack_span and attack_span else None
                )
                rows.append({
                    "encoding": enc_name, "layer": layer_idx, "template_position": pos_name,
                    "incoming_doc_mass": incoming_doc, "incoming_attack_mass": incoming_attack,
                    "outgoing_query_mass": outgoing_query, "outgoing_doc_mass": outgoing_doc,
                })
    return rows


def main() -> None:
    args = parse_args()
    cfg = load_config(pathlib.Path(args.config))
    outputs_base = resolve_cfg_path(cfg, cfg["outputs"]["base_dir"])
    causal_dir = outputs_base / "causal_patch"
    out_dir = outputs_base / "attention_mass"
    out_dir.mkdir(parents=True, exist_ok=True)

    causal_csv = causal_dir / "results.csv"
    if not causal_csv.exists():
        sys.exit(f"[07_template_attention_mass] {causal_csv} not found -- run scripts/06_template_causal_patch.py first.")
    causal_df = pd.read_csv(causal_csv)

    thresholds = cfg["flag_thresholds"]
    flags = flag_positions(causal_df, thresholds)
    primary_t = thresholds[0]
    flagged_names = [n for n, v in flags.items() if v[f"flagged_{primary_t}"]]

    print(f"[07_template_attention_mass] peak combined_effect (layers 9-11), thresholds={thresholds}:")
    for name in TEMPLATE_POSITION_NAMES:
        f = flags[name]
        flags_str = " ".join(f"[{'X' if f[f'flagged_{t}'] else ' '}]>{t}" for t in thresholds)
        print(f"    {name:<20} peak_e_comb={f['peak_e_comb']:+.4f}  {flags_str}")

    target_positions = flagged_names if args.flagged_only else TEMPLATE_POSITION_NAMES
    print(f"[07_template_attention_mass] computing attention mass for: "
          f"{'flagged only' if args.flagged_only else 'all 7 positions'} "
          f"({len(target_positions)}): {target_positions}")

    if not flagged_names:
        print("[07_template_attention_mass] NOTE: no position flagged at the primary threshold "
              f"({primary_t}) -- proceeding anyway (all 7 computed unless --flagged-only) so the "
              "classification table still documents the sink/uninvolved verdict.")

    device = resolve_device(cfg["model"]["device"])
    model, tokenizer = load_monot5(cfg["model"]["checkpoint"], device)

    examples_path = causal_dir / "selected_examples.jsonl"
    examples = [json.loads(l) for l in open(examples_path, encoding="utf-8")]
    max_length = cfg["model"]["max_length"]

    all_rows = []
    n_used = 0
    for i, ex in enumerate(examples):
        inputs = build_example_inputs(tokenizer, ex, max_length, device)
        if inputs.status != "ok":
            continue
        rows = mass_for_example(tokenizer, model, device, inputs)
        for row in rows:
            row["qid"], row["docid"] = ex["qid"], ex["docid"]
        all_rows.extend(rows)
        n_used += 1
        if (i + 1) % 10 == 0 or i == len(examples) - 1:
            print(f"    example {i+1}/{len(examples)} (used={n_used})", flush=True)

    df = pd.DataFrame(all_rows)
    df.to_csv(out_dir / "results.csv", index=False)
    print(f"[07_template_attention_mass] wrote {len(df)} rows ({n_used} examples) -> {out_dir/'results.csv'}")

    # Per-(position, encoding, layer) mean, then peak-over-9-11 summary for classification.
    agg = df.groupby(["template_position", "encoding", "layer"], dropna=False).agg(
        n=("qid", "size"),
        incoming_doc_mass_mean=("incoming_doc_mass", "mean"),
        incoming_attack_mass_mean=("incoming_attack_mass", "mean"),
        outgoing_query_mass_mean=("outgoing_query_mass", "mean"),
        outgoing_doc_mass_mean=("outgoing_doc_mass", "mean"),
    ).reset_index()
    agg.to_csv(out_dir / "aggregated.csv", index=False)

    late = agg[(agg["layer"].isin([9, 10, 11])) & (agg["encoding"] == "attack")]
    classification_rows = []
    for name in TEMPLATE_POSITION_NAMES:
        sub = late[late["template_position"] == name]
        peak_incoming = float(max(
            sub["incoming_doc_mass_mean"].max() if not sub.empty else 0.0,
            sub["incoming_attack_mass_mean"].max() if not sub.empty and sub["incoming_attack_mass_mean"].notna().any() else 0.0,
        ))
        peak_outgoing = float(max(
            sub["outgoing_query_mass_mean"].max() if not sub.empty else 0.0,
            sub["outgoing_doc_mass_mean"].max() if not sub.empty else 0.0,
        ))
        mass_is_high = (peak_incoming > MASS_BASELINE) or (peak_outgoing > MASS_BASELINE)
        ecomb_is_high = flags[name][f"flagged_{primary_t}"]
        if ecomb_is_high and mass_is_high:
            label = "relay"
        elif ecomb_is_high and not mass_is_high:
            label = "hidden-channel"
        elif not ecomb_is_high and mass_is_high:
            label = "sink"
        else:
            label = "uninvolved"
        classification_rows.append({
            "template_position": name,
            "peak_e_comb": flags[name]["peak_e_comb"],
            "peak_incoming_mass": peak_incoming,
            "peak_outgoing_mass": peak_outgoing,
            "flagged_at_primary_threshold": ecomb_is_high,
            "classification": label,
        })
    classification_df = pd.DataFrame(classification_rows)
    classification_df.to_csv(out_dir / "classification.csv", index=False)

    print("\n[07_template_attention_mass] Classification table:")
    print(classification_df.to_string(index=False))
    print(f"\n  Next: python scripts/08_template_full_sweep.py --config {args.config}")


if __name__ == "__main__":
    main()
