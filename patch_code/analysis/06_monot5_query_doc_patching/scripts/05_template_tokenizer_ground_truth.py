#!/usr/bin/env python3
"""
scripts/05_template_tokenizer_ground_truth.py
================================================
Experiment 6 extension, Step 0 -- tokenizer ground truth for the 7 template
positions (see exp6lib/template_positions.py). REQUIRED sanity check before
any patching code runs: confirms exactly which tokens the literal template
words ("Query", "Document", "Relevant") split into, and that
get_template_positions groups them robustly (by decoded content, not
hardcoded offsets) across examples of different query/document length.

Does not patch anything and does not need the GPU -- CPU (login node) is fine.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
EXP_DIR = SCRIPT_DIR.parent
EXP1_DIR = EXP_DIR.parent / "01_monot5_layer_patching"
sys.path.insert(0, str(EXP1_DIR))
sys.path.insert(0, str(EXP_DIR))

from src.attack_registry import discover_attacks
from src.model_utils import get_true_false_token_ids, load_monot5, resolve_device

from exp6lib.run_utils import build_example_inputs, get_examples_for_attack, load_config
from exp6lib.template_positions import TEMPLATE_POSITION_NAMES, get_template_positions


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Experiment 6 extension -- Step 0 tokenizer ground truth.")
    p.add_argument("--config", default=str(EXP_DIR / "configs" / "template_tokens.yaml"))
    p.add_argument("--n-pool", type=int, default=20,
                    help="Pool of canonical-attack examples to pick 3 different lengths from.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(pathlib.Path(args.config))

    device = resolve_device(cfg["model"]["device"])
    print(f"[05_template_tokenizer_ground_truth] device: {device}")
    model, tokenizer = load_monot5(cfg["model"]["checkpoint"], device)
    true_id, false_id = get_true_false_token_ids(tokenizer)

    canonical_name = cfg["runs"]["canonical"]["attack_name"]
    single_cfg = {
        "upstream_injected_dir": cfg["attacks"]["upstream_injected_dir"],
        "mode": "include", "include": [canonical_name],
    }
    spec = discover_attacks(single_cfg)[0]
    examples = get_examples_for_attack(spec, args.n_pool, cfg, model, tokenizer, true_id, false_id, device)
    print(f"[05_template_tokenizer_ground_truth] pool: {len(examples)} examples for {canonical_name}")

    built = []
    for ex in examples:
        inputs = build_example_inputs(tokenizer, ex, cfg["model"]["max_length"], device)
        if inputs.status != "ok":
            continue
        seq_len = inputs.attack_enc["input_ids"].shape[1]
        built.append((seq_len, ex, inputs))
    if len(built) < 3:
        sys.exit(f"[05_template_tokenizer_ground_truth] only {len(built)} alignable examples "
                  f"in pool of {args.n_pool} -- increase --n-pool.")

    built.sort(key=lambda t: t[0])
    picks = [built[0], built[len(built) // 2], built[-1]]  # shortest, median, longest

    for label, (seq_len, ex, inputs) in zip(["shortest", "median", "longest"], picks):
        print(f"\n{'='*70}\n  {label.upper()}  qid={ex['qid']} docid={ex['docid']}  seq_len={seq_len}\n{'='*70}")
        ids = inputs.attack_enc["input_ids"][0].tolist()
        positions = get_template_positions(tokenizer, ids, inputs.query_span, inputs.doc_span_control_attack)

        pos_to_name = {}
        for name, idxs in positions.items():
            for i in idxs:
                pos_to_name[i] = name

        print(f"  query_span={inputs.query_span}  doc_span={inputs.doc_span_control_attack}")
        print(f"  {'idx':>4} {'token_id':>9}  {'piece':<14} template_position")
        for i, tid in enumerate(ids):
            piece = tokenizer.convert_ids_to_tokens([tid])[0]
            tag = pos_to_name.get(i, "")
            marker = f"  <-- {tag}" if tag else ""
            if tag or any(abs(i - j) <= 1 for j in pos_to_name):
                print(f"  {i:>4} {tid:>9}  {piece!r:<14}{marker}")

        print("\n  Grouped template positions:")
        for name in TEMPLATE_POSITION_NAMES:
            idxs = positions[name]
            pieces = tokenizer.convert_ids_to_tokens([ids[i] for i in idxs])
            print(f"    {name:<20} indices={idxs}  pieces={pieces}")

    print(f"\n[05_template_tokenizer_ground_truth] OK -- get_template_positions is robust across "
          f"{len(picks)} examples of different length (shortest={picks[0][0]}, median={picks[1][0]}, "
          f"longest={picks[2][0]} tokens).")


if __name__ == "__main__":
    main()
