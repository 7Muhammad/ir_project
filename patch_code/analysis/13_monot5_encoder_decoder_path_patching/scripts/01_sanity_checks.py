#!/usr/bin/env python3
"""
scripts/01_sanity_checks.py
=============================
Required validation tests (task spec "SANITY CHECKS") run on ONE real
(attack, example) before the full 558-path sweep is launched. Exits
non-zero if any check fails.

  1. No-change control        -- hybrid == baseline => path effect ~ 0.
  2. Sender-only reconstruction -- "whole decoder reads hybrid" matches
                                     Experiment 11's own per-head patch.
  3. Full encoder control      -- unpatched decoder reproduces
                                     score_control / score_attack exactly.
  4. Single-receiver isolation -- Step D's scored decoder pass never
                                     swaps key_value_states anywhere (the
                                     only substitution is the O-projection
                                     block patch at one head, one layer).
  5. Correct head sets         -- 18 senders, 31 receivers, 558 paths.
  6. Successful-instance filter -- a near-zero-delta example is skipped.
  7. Sample cap                -- selection never exceeds max_examples_per_attack.
  8. Forward/reverse baseline  -- forward path always starts from the
     preservation                 control encoder state, reverse from attack.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import torch

EXP_DIR = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(EXP_DIR))

from exp13lib.head_lists import (  # noqa: E402
    N_PATHS_EXPECTED,
    N_RECEIVERS_EXPECTED,
    N_SENDERS_EXPECTED,
    ReceiverHead,
    SenderHead,
    assert_head_sets,
    load_receivers,
    load_senders,
)
from exp13lib.path_engine import (  # noqa: E402
    SKIP_EPSILON,
    build_encoder_hybrid,
    cache_encoder_and_score,
    compute_receiver_layer_step_b_cache,
    cross_attn_kv_swap_pre_hook,
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

from headlib.engine import decoder_pass_scores, head_scores_for_slot  # noqa: E402
import headlib.head_hooks as dec_hooks  # noqa: E402
import exp11lib.head_hooks as enc_hooks  # noqa: E402
import exp11lib.engine as enc_engine  # noqa: E402

TOL = 1e-4
results_log = []


def check(name: str, passed: bool, detail: str = "") -> None:
    status = "PASS" if passed else "FAIL"
    line = f"[{status}] {name}" + (f" -- {detail}" if detail else "")
    print(line)
    results_log.append(line)
    if not passed:
        global any_failed
        any_failed = True


any_failed = False


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Experiment 13 -- required sanity checks.")
    p.add_argument("--config", default=str(EXP_DIR / "configs" / "default.yaml"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(pathlib.Path(args.config))
    dbg = cfg["debug"]

    model, tokenizer, true_id, false_id, device = load_model(cfg)

    # ---- Check 5: correct head sets ----
    senders = load_senders()
    receivers = load_receivers()
    try:
        assert_head_sets(senders, receivers)
        check("5. correct head sets (18 senders, 31 receivers, 558 paths)", True,
              f"{len(senders)} x {len(receivers)} = {len(senders)*len(receivers)}")
    except AssertionError as e:
        check("5. correct head sets", False, str(e))

    # ---- Real example to run checks 1-4, 6, 8 on ----
    attacks = {s.attack_name: s for s in get_attacks(cfg)}
    spec = attacks[dbg["attack_name"]]
    reuse_dir = resolve_cfg_path(cfg, cfg["selection"]["reuse_dir"])
    examples = select_examples_for_attack(spec.attack_name, reuse_dir, 1, cfg["runtime"]["seed"])
    if not examples:
        sys.exit(f"[sanity] no qualifying examples for {spec.attack_name}; cannot run checks.")
    example = examples[0]
    control_enc, attack_enc, align_result = build_encodings(tokenizer, example, cfg["model"]["max_length"], device)
    if align_result.status != "ok":
        sys.exit(f"[sanity] alignment failed for the debug example: {align_result}")

    sender = SenderHead(layer=dbg["sender_layer"], head_idx=dbg["sender_head"], label="debug_sender")
    receiver = ReceiverHead(layer=dbg["receiver_layer"], head_idx=dbg["receiver_head"],
                             component="decoder_cross_attn", label="debug_receiver")

    sender_layers = [sender.layer]
    ctrl_o_cache, ctrl_hidden, ctrl_mask, score_control = cache_encoder_and_score(
        model, control_enc, device, sender_layers, true_id, false_id
    )
    atk_o_cache, atk_hidden, atk_mask, score_attack = cache_encoder_and_score(
        model, attack_enc, device, sender_layers, true_id, false_id
    )
    delta = score_attack - score_control
    n_heads, d_kv, _ = enc_hooks.head_geometry(model)
    full_mask = enc_hooks.block_diag_head_mask(n_heads, d_kv, device)
    single_mask = full_mask[sender.head_idx : sender.head_idx + 1]

    # ---- Check 1: no-change control ----
    # "hybrid" built from the BASE run's own cached activation at that layer
    # (control->control for forward, attack->attack for reverse): must be a
    # true no-op, so the isolated-path score must equal the baseline score
    # (not merely path effect ~ 0 -- the SCORE itself must be unchanged).
    noop_hybrid_hidden, noop_hybrid_mask = build_encoder_hybrid(
        model, control_enc, ctrl_o_cache[enc_hooks.slot_key(sender.layer)], sender.layer, single_mask, device
    )
    step_b_noop = compute_receiver_layer_step_b_cache(
        model, ctrl_hidden, ctrl_mask, noop_hybrid_hidden, receiver.layer, true_id, false_id, device
    )
    dec_n_heads, dec_d_kv, _ = dec_hooks.head_geometry(model)
    dec_full_mask = dec_hooks.block_diag_head_mask(dec_n_heads, dec_d_kv, device)
    noop_scores = head_scores_for_slot(
        model, ctrl_hidden, ctrl_mask, "decoder_cross_attn", receiver.layer, step_b_noop, dec_full_mask,
        true_id, false_id,
    )
    noop_score = noop_scores[receiver.head_idx]
    noop_effect = (noop_score - score_control) / delta if abs(delta) > 1e-9 else float("nan")
    check(
        "1. no-change control (hybrid == baseline => path effect ~ 0)",
        abs(noop_score - score_control) < TOL and abs(noop_effect) < 1e-2,
        f"score diff={noop_score - score_control:.6f}, path effect={noop_effect:.6f}",
    )

    # ---- Check 2: sender-only reconstruction ----
    hybrid_fwd_hidden, hybrid_fwd_mask = build_encoder_hybrid(
        model, control_enc, atk_o_cache[enc_hooks.slot_key(sender.layer)], sender.layer, single_mask, device
    )
    sender_only_score = decoder_pass_scores(model, hybrid_fwd_hidden, hybrid_fwd_mask, true_id, false_id, batch_rows=1)[0].item()

    # Reference: Experiment 11's own per-head forward-patch score for this sender.
    ref_scores = enc_engine.head_scores_for_layer(
        model, control_enc, device, sender.layer, atk_o_cache[enc_hooks.slot_key(sender.layer)], full_mask,
        true_id, false_id,
    )
    ref_score = ref_scores[sender.head_idx]
    check(
        "2. sender-only reconstruction matches Experiment 11's per-head patch",
        abs(sender_only_score - ref_score) < TOL,
        f"path-engine score={sender_only_score:.6f} vs exp11 score={ref_score:.6f}",
    )

    # ---- Check 3: full encoder control reproduces score_control / score_attack ----
    plain_ctrl_score = decoder_pass_scores(model, ctrl_hidden, ctrl_mask, true_id, false_id, batch_rows=1)[0].item()
    plain_atk_score = decoder_pass_scores(model, atk_hidden, atk_mask, true_id, false_id, batch_rows=1)[0].item()
    check(
        "3. full (unpatched) encoder control reproduces score_control/score_attack",
        abs(plain_ctrl_score - score_control) < TOL and abs(plain_atk_score - score_attack) < TOL,
        f"control diff={plain_ctrl_score - score_control:.6f}, attack diff={plain_atk_score - score_attack:.6f}",
    )

    # ---- Check 4 & 8: single-receiver isolation + forward/reverse baseline preservation ----
    # Instrument every decoder cross-attention layer's key_value_states during
    # the SCORED Step-D pass; none may ever be swapped away from the plain
    # baseline (ctrl_hidden for forward, atk_hidden for reverse) -- the only
    # substitution mechanism active during a scored pass is the existing,
    # already-tested O-projection block patch at one head of one layer.
    seen = {}

    def make_observer(layer_idx):
        def hook(module, args, kwargs):
            seen[layer_idx] = kwargs.get("key_value_states")
            return args, kwargs
        return hook

    handles = [
        model.decoder.block[L].layer[1].register_forward_pre_hook(make_observer(L), with_kwargs=True)
        for L in range(len(model.decoder.block))
    ]
    try:
        step_b_fwd = compute_receiver_layer_step_b_cache(
            model, ctrl_hidden, ctrl_mask, hybrid_fwd_hidden, receiver.layer, true_id, false_id, device
        )
        seen.clear()
        head_scores_for_slot(
            model, ctrl_hidden, ctrl_mask, "decoder_cross_attn", receiver.layer, step_b_fwd, dec_full_mask,
            true_id, false_id,
        )
        # head_scores_for_slot broadcasts the single-row baseline to a
        # dec_n_heads-row batch (one row per hypothetical head patch), so
        # compare against the expanded baseline, not the raw (1, L, d) tensor.
        forward_ok = all(torch.equal(seen[L], ctrl_hidden.expand_as(seen[L])) for L in seen)
    finally:
        for h in handles:
            h.remove()

    hybrid_rev_hidden, _ = build_encoder_hybrid(
        model, attack_enc, ctrl_o_cache[enc_hooks.slot_key(sender.layer)], sender.layer, single_mask, device
    )
    handles = [
        model.decoder.block[L].layer[1].register_forward_pre_hook(make_observer(L), with_kwargs=True)
        for L in range(len(model.decoder.block))
    ]
    try:
        step_b_rev = compute_receiver_layer_step_b_cache(
            model, atk_hidden, atk_mask, hybrid_rev_hidden, receiver.layer, true_id, false_id, device
        )
        seen.clear()
        head_scores_for_slot(
            model, atk_hidden, atk_mask, "decoder_cross_attn", receiver.layer, step_b_rev, dec_full_mask,
            true_id, false_id,
        )
        reverse_ok = all(torch.equal(seen[L], atk_hidden.expand_as(seen[L])) for L in seen)
    finally:
        for h in handles:
            h.remove()

    check(
        "4. single-receiver isolation (Step D never swaps key_value_states anywhere)",
        forward_ok and reverse_ok,
        f"forward_ok={forward_ok}, reverse_ok={reverse_ok}",
    )
    check(
        "8. forward/reverse baseline preservation (forward starts from control, reverse from attack)",
        forward_ok and reverse_ok,
        "same evidence as check 4: Step D's decoder base is always ctrl_hidden (forward) / atk_hidden (reverse)",
    )

    # ---- Check 6: successful-instance filtering ----
    degenerate_meta = {"attack_name": spec.attack_name, "qid": "degenerate", "docid": "degenerate", "sample_index": 0, "seed": 0}
    degenerate_result = run_example_paths(
        model, control_enc, control_enc, [sender], [receiver], true_id, false_id, device, degenerate_meta,
    )
    check(
        "6. successful-instance filtering (delta > 1e-4 enforced)",
        degenerate_result is None,
        "control_enc vs itself (delta=0) correctly skipped" if degenerate_result is None else "FAILED TO SKIP a zero-delta example",
    )

    # ---- Check 7: sample cap ----
    capped = select_examples_for_attack(spec.attack_name, reuse_dir, 10, cfg["runtime"]["seed"])
    check("7. sample cap (<= 10 examples for the initial breadth run)", len(capped) <= 10, f"got {len(capped)}")

    print()
    if any_failed:
        print("SANITY CHECKS: at least one check FAILED. Do not launch the full sweep.")
        out_dir = resolve_cfg_path(cfg, cfg["outputs"]["diagnostics_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "sanity_check_report.txt").write_text("\n".join(results_log) + "\n", encoding="utf-8")
        sys.exit(1)
    else:
        print("SANITY CHECKS: all passed.")
        out_dir = resolve_cfg_path(cfg, cfg["outputs"]["diagnostics_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "sanity_check_report.txt").write_text("\n".join(results_log) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
