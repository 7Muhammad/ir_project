#!/usr/bin/env python3
"""
scripts/00_debug_single_path.py
=================================
Lightweight debug configuration: 1 attack, 1 example, 1 encoder sender, 1
decoder receiver. Run this BEFORE the full 558-path sweep (per task spec).

Prints baseline scores, the ordinary "sender-only" patched score (all
decoder heads implicitly reading the hybrid encoder state -- Experiment
11's own patch, reproduced via this experiment's hybrid-construction code
path as a cross-check), the isolated single-receiver path score, and the
forward / reverse / combined path effects.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import torch

EXP_DIR = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(EXP_DIR))

from exp13lib.head_lists import ReceiverHead, SenderHead  # noqa: E402
from exp13lib.path_engine import (  # noqa: E402
    build_encoder_hybrid,
    cache_encoder_and_score,
    run_example_paths,
)
from exp13lib.run_utils import (  # noqa: E402
    build_encodings,
    get_attacks,
    load_config,
    load_model,
    resolve_cfg_path,
    select_examples_for_attack,
)

from headlib.engine import decoder_pass_scores  # noqa: E402
import headlib.head_hooks as dec_hooks  # noqa: E402
import exp11lib.head_hooks as enc_hooks  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Experiment 13 -- single-path debug run.")
    p.add_argument("--config", default=str(EXP_DIR / "configs" / "default.yaml"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(pathlib.Path(args.config))
    dbg = cfg["debug"]

    model, tokenizer, true_id, false_id, device = load_model(cfg)

    attacks = {s.attack_name: s for s in get_attacks(cfg)}
    if dbg["attack_name"] not in attacks:
        sys.exit(f"[debug] attack '{dbg['attack_name']}' not found in the 105-attack grid.")
    spec = attacks[dbg["attack_name"]]

    reuse_dir = resolve_cfg_path(cfg, cfg["selection"]["reuse_dir"])
    examples = select_examples_for_attack(spec.attack_name, reuse_dir, dbg["n_examples"], cfg["runtime"]["seed"])
    if not examples:
        sys.exit(f"[debug] no qualifying (delta > 1e-4) examples found for {spec.attack_name}.")
    example = examples[0]
    print(f"[debug] attack={spec.attack_name} qid={example['qid']} docid={example['docid']}")

    control_enc, attack_enc, align_result = build_encodings(tokenizer, example, cfg["model"]["max_length"], device)
    if align_result.status != "ok":
        sys.exit(f"[debug] alignment failed: {align_result}")

    sender = SenderHead(layer=dbg["sender_layer"], head_idx=dbg["sender_head"], label=f"L{dbg['sender_layer']}H{dbg['sender_head']}")
    receiver = ReceiverHead(
        layer=dbg["receiver_layer"], head_idx=dbg["receiver_head"], component="decoder_cross_attn",
        label=f"L{dbg['receiver_layer']}-X-H{dbg['receiver_head']}",
    )
    meta = {"attack_name": spec.attack_name, "qid": example["qid"], "docid": example["docid"], "sample_index": 0, "seed": cfg["runtime"]["seed"]}

    result = run_example_paths(model, control_enc, attack_enc, [sender], [receiver], true_id, false_id, device, meta)
    if result is None:
        sys.exit("[debug] example skipped: |delta| < SKIP_EPSILON (should not happen for a reused example).")

    row = result["rows"][0]
    print(f"[debug] score_control        = {result['score_control']:.6f}")
    print(f"[debug] score_attack         = {result['score_attack']:.6f}")
    print(f"[debug] delta                = {result['delta']:.6f}")
    print(f"[debug] sender               = {sender.label}   receiver = {receiver.label}")
    print(f"[debug] score_path_forward   = {row['score_path_forward']:.6f}")
    print(f"[debug] score_path_reverse   = {row['score_path_reverse']:.6f}")
    print(f"[debug] path_forward         = {row['path_forward']:.6f}")
    print(f"[debug] path_reverse         = {row['path_reverse']:.6f}")
    print(f"[debug] path_combined        = {row['path_combined']:.6f}")

    # ---- Cross-check: ordinary "sender-only" reconstruction ----
    # Build encoder_hybrid_forward(A) directly and score it with the WHOLE
    # decoder reading it (no receiver isolation at all) -- this must equal
    # Experiment 11's own per-head forward-patch score for sender A.
    sender_layers = [sender.layer]
    ctrl_o_cache, ctrl_hidden, ctrl_mask, score_control2 = cache_encoder_and_score(
        model, control_enc, device, sender_layers, true_id, false_id
    )
    atk_o_cache, atk_hidden, atk_mask, score_attack2 = cache_encoder_and_score(
        model, attack_enc, device, sender_layers, true_id, false_id
    )
    n_heads, d_kv, _ = enc_hooks.head_geometry(model)
    full_mask = enc_hooks.block_diag_head_mask(n_heads, d_kv, device)
    single_mask = full_mask[sender.head_idx : sender.head_idx + 1]
    hybrid_hidden, hybrid_mask = build_encoder_hybrid(
        model, control_enc, atk_o_cache[enc_hooks.slot_key(sender.layer)], sender.layer, single_mask, device
    )
    sender_only_score = decoder_pass_scores(model, hybrid_hidden, hybrid_mask, true_id, false_id, batch_rows=1)[0].item()
    sender_only_effect = (sender_only_score - score_control2) / (score_attack2 - score_control2)
    print(f"[debug] sender-only (whole decoder reads hybrid) score = {sender_only_score:.6f}, "
          f"fwd_effect = {sender_only_effect:.6f}  (cross-check vs Experiment 11's own per-head patch)")

    print("[debug] OK -- run scripts/01_sanity_checks.py next before launching the full sweep.")


if __name__ == "__main__":
    main()
