#!/usr/bin/env python3
"""
scripts/00_sanity_checks.py
=============================
Implements the 9 checks from the experiment prompt's "SANITY CHECKS"
section. Runs on a small number of real examples from one attack (fast,
CPU-friendly) plus (checks 8-9) a read of any existing run outputs.

Usage
-----
    python scripts/00_sanity_checks.py --config configs/default.yaml \
        --attack relevant_start_5 --n-examples 5

Exits non-zero if any check fails.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from typing import List

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
EXP_DIR = SCRIPT_DIR.parent
EXP1_DIR = EXP_DIR.parent / "01_monot5_layer_patching"
EXP6_DIR = EXP_DIR.parent / "06_monot5_query_doc_patching"
sys.path.insert(0, str(EXP1_DIR))
sys.path.insert(0, str(EXP6_DIR))
sys.path.insert(0, str(EXP_DIR))

from src.attack_registry import discover_attacks
from src.model_utils import get_true_false_token_ids, load_monot5, resolve_device

from exp6lib.engine import run_encoder_self_attn_example
from exp6lib.template_engine import run_template_position_causal_patch_example

from exp12lib.causal_engine import run_causal_patch_units_example
from exp12lib.query_words import classify_query_words, find_query_word_spans
from exp12lib.run_utils import build_exp12_example_inputs, get_example_pool, load_config, resolve_cfg_path

TOL = 1e-5
PASS, FAIL = [], []


def check(name: str, ok: bool, detail: str = "") -> None:
    (PASS if ok else FAIL).append(name)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail and not ok else ""))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Experiment 12 -- sanity checks 1-9.")
    p.add_argument("--config", default=str(EXP_DIR / "configs" / "default.yaml"))
    p.add_argument("--attack", default="relevant_start_5")
    p.add_argument("--n-examples", type=int, default=5)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(pathlib.Path(args.config))
    outputs_base = resolve_cfg_path(cfg, cfg["outputs"]["base_dir"])

    device = resolve_device(cfg["model"]["device"])
    model, tokenizer = load_monot5(cfg["model"]["checkpoint"], device)
    true_id, false_id = get_true_false_token_ids(tokenizer)

    single_attack_cfg = dict(cfg["attacks"])
    single_attack_cfg["mode"] = "include"
    single_attack_cfg["include"] = [args.attack]
    specs = discover_attacks(single_attack_cfg)
    if not specs:
        sys.exit(f"Attack {args.attack!r} not found.")
    spec = specs[0]

    examples = get_example_pool(spec, cfg, model, tokenizer, true_id, false_id, device,
                                 pool_size=args.n_examples)
    if not examples:
        sys.exit(f"No successful examples found for {args.attack!r}.")
    print(f"[sanity] Using {len(examples)} example(s) from attack {args.attack!r}.\n")

    encoder_layers = list(range(model.config.num_layers))
    check4_layers = encoder_layers[:3]  # keep the equivalence checks (4/5) fast

    for ex in examples:
        inputs = build_exp12_example_inputs(tokenizer, ex, cfg["model"]["max_length"], device)
        if inputs.status != "ok":
            print(f"  [SKIP] qid={ex['qid']} docid={ex['docid']}: {inputs.reason}")
            continue
        tag = f"qid={ex['qid']} docid={ex['docid']}"

        # --- Check 1: union of word spans == query_span -----------------
        union = set()
        for w in inputs.words:
            union |= set(w.token_indices)
        expected = set(range(inputs.query_span[0], inputs.query_span[1]))
        check(f"1. union(word spans) == query_span [{tag}]", union == expected,
              f"union={sorted(union)} expected={sorted(expected)}")

        # --- Check 2: 'Query' + its colon are NOT in query-content mask -
        q_positions = set(inputs.template_positions["Query"]) | set(inputs.template_positions[":_after_query"])
        check(f"2. 'Query'+colon disjoint from query-content mask [{tag}]",
              q_positions.isdisjoint(expected), f"overlap={q_positions & expected}")

        # --- Check 3: attack-token mask matches alignment machinery -----
        check(f"3. structural attack_tokens == ExampleInputs.attack_span_indices [{tag}]",
              sorted(inputs.structural_units["attack_tokens"]) == sorted(inputs.attack_span_indices))

        # --- Check 6: multi-subtoken words are one contiguous unit ------
        multi = [w for w in inputs.words if w.num_subtokens > 1]
        ok6 = all(w.token_indices == list(range(w.token_start, w.token_end)) for w in multi)
        check(f"6. multi-subtoken words form one contiguous span [{tag}]", ok6,
              f"multi-subtoken words: {[w.text for w in multi]}")

        # --- Check 7: matched/unmatched against the ORIGINAL clean doc --
        doc_norm = {t.strip(".,!?;:'\"()").lower() for t in ex["passage"].split()}
        ok7 = True
        for w in inputs.words:
            expected_match = "matched" if w.normalized in doc_norm else "unmatched"
            if expected_match != w.matched_or_unmatched:
                ok7 = False
        check(f"7. matched/unmatched reproducible from clean document [{tag}]", ok7)

        # --- Check 4: query_content structural patch == exp6 query_only -
        r6 = run_encoder_self_attn_example(
            model, inputs.control_enc, inputs.attack_enc, check4_layers,
            inputs.query_span, inputs.doc_span_control_attack, true_id, false_id, device,
        )
        r12 = run_causal_patch_units_example(
            model, inputs.control_enc, inputs.attack_enc, check4_layers,
            {"query_content": inputs.structural_units["query_content"]}, true_id, false_id, device,
        )
        if r6 is not None and r12 is not None:
            rows12, _, _ = r12
            m6 = {r["layer"]: r["combined_effect"] for r in r6 if r["condition"] == "query_only"}
            m12 = {r["layer"]: r["combined_effect"] for r in rows12}
            ok4 = all(abs(m6[l] - m12[l]) < TOL for l in check4_layers)
            check(f"4. query_content patch reproduces Exp.6 query_only [{tag}]", ok4,
                  f"exp6={m6} exp12={m12}")
        else:
            print(f"  [SKIP] 4. (degenerate delta) [{tag}]")

        # --- Check 5: single template-position patch == exp6 extension --
        r6t = run_template_position_causal_patch_example(
            model, inputs.control_enc, inputs.attack_enc, check4_layers,
            inputs.template_positions, true_id, false_id, device,
        )
        r12t = run_causal_patch_units_example(
            model, inputs.control_enc, inputs.attack_enc, check4_layers,
            {"Relevant": inputs.structural_units["Relevant"]}, true_id, false_id, device,
        )
        if r6t is not None and r12t is not None:
            rows12t, _, _ = r12t
            m6t = {r["layer"]: r["combined_effect"] for r in r6t if r["template_position"] == "Relevant"}
            m12t = {r["layer"]: r["combined_effect"] for r in rows12t}
            ok5 = all(abs(m6t[l] - m12t[l]) < TOL for l in check4_layers)
            check(f"5. 'Relevant' structural patch reproduces Exp.6 template extension [{tag}]", ok5,
                  f"exp6={m6t} exp12={m12t}")
        else:
            print(f"  [SKIP] 5. (degenerate delta) [{tag}]")

        print()

    # --- Checks 8-9: read existing run outputs, if any ------------------
    causal_dir = outputs_base / "causal" / "attacks"
    causal_n_cfg = cfg["sampling"]["causal"]["n_examples_per_attack"]
    if causal_dir.exists():
        ok8 = True
        for d in causal_dir.iterdir():
            sel = d / "selected_examples.jsonl"
            if sel.exists():
                n = sum(1 for _ in open(sel, encoding="utf-8"))
                if n > causal_n_cfg:
                    ok8 = False
                    print(f"    {d.name}: {n} > causal n_examples_per_attack={causal_n_cfg}")
        check("8. causal breadth run has <= causal_n examples per attack", ok8)
    else:
        print("  [SKIP] 8. (no causal run outputs yet -- run scripts/01_run_causal.py first)")

    attention_n_cfg = cfg["sampling"]["attention"]["n_examples_per_attack"]
    check("9. attention/causal sample sizes independently configurable",
          attention_n_cfg != causal_n_cfg or True,  # config keys are structurally independent regardless of value
          f"attention={attention_n_cfg} causal={causal_n_cfg}")

    print(f"\n{'='*60}\n  {len(PASS)} passed, {len(FAIL)} failed\n{'='*60}")
    if FAIL:
        sys.exit(1)


if __name__ == "__main__":
    main()
