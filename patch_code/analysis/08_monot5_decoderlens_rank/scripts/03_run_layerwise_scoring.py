#!/usr/bin/env python3
"""
scripts/03_run_layerwise_scoring.py
===================================
DecoderLens layerwise scoring for an attack.

For every target example and every variant (original / padded_control / attack)
we compute a monoT5 score at each encoder layer by forcing the decoder to
cross-attend to that layer's hidden states:

    score_l = logit("true") - logit("false")

We also compute the CLEAN layerwise scores of every candidate in each used
query's pool (needed by Stage 04 to rank the target against the whole set).

Final-layer sanity check
------------------------
For each attack the DecoderLens score at the top encoder layer must closely
match the normal monoT5 score.  We compare them on the attack variant and save
outputs/attacks/{attack_name}/sanity/final_layer_equivalence.json.

Outputs
-------
  scores/layerwise_target_scores.csv
  scores/layerwise_candidate_scores.parquet   (clean candidate scores)
  sanity/final_layer_equivalence.json

Usage
-----
  python scripts/03_run_layerwise_scoring.py --config configs/default.yaml \
      --attack-name relevant_start_5
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Dict, List

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import torch

from src.attack_registry import AttackSpec, discover_attacks
from src.data_loading import build_candidate_sets, load_attack_tsv
from src.decoderlens import (
    get_true_false_token_ids,
    load_monot5,
    normal_monot5_score,
    num_encoder_layers,
    resolve_device,
)
from src.padded_control import build_variant_encodings, encode_clean
from src.scoring import score_candidates_layerwise, score_target_variant_layerwise
from src.utils import attack_output_dirs, layer_name, load_config


def _load_target_pairs(pairs_path: pathlib.Path) -> List[Dict]:
    pairs: List[Dict] = []
    with open(pairs_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))
    return pairs


def score_attack(
    spec: AttackSpec,
    cfg: dict,
    model,
    tokenizer,
    true_id: int,
    false_id: int,
    device: torch.device,
    base_dir: pathlib.Path,
) -> Dict:
    """Run layerwise scoring + sanity check for one attack.  Returns a counts dict."""
    out_dirs = attack_output_dirs(base_dir, spec.attack_name)
    pairs_path = out_dirs["pairs"] / "target_pairs.jsonl"
    if not pairs_path.exists():
        raise FileNotFoundError(
            f"Missing {pairs_path}. Run 02_prepare_decoderlens_pairs.py first."
        )
    targets = _load_target_pairs(pairs_path)
    if not targets:
        print(f"[03] {spec.attack_name}: no aligned targets, nothing to score.")
        return {"n_examples_used": 0}

    max_length = int(cfg["model"]["max_length"])
    aln = bool(cfg["decoderlens"]["apply_final_layer_norm"])
    candidate_top_k = int(cfg["ranking"]["candidate_top_k"])
    sel = cfg["selection"]
    mode = sel.get("mode", "all_aligned")
    max_targets = int(sel.get("max_target_examples_per_attack", 100))
    min_delta = float(sel.get("min_attack_delta_final_layer", 0.0))
    n_layers = num_encoder_layers(model)

    # --- Candidate clean scores for every used query (computed once) ---------
    used_qids = sorted({t["qid"] for t in targets})
    records = load_attack_tsv(spec.path)
    candidate_sets = build_candidate_sets(records, candidate_top_k)

    candidate_scores: Dict[str, Dict[str, List[float]]] = {}
    for qid in used_qids:
        cands = candidate_sets.get(qid, [])
        candidate_scores[qid] = score_candidates_layerwise(
            model, tokenizer, cands, true_id, false_id, max_length, device, aln
        )
        print(f"[03] {spec.attack_name}: scored {len(cands)} candidates for qid={qid}")

    # --- Per-target variant scores ------------------------------------------
    target_rows: List[Dict] = []
    final_layer_diffs: List[float] = []
    per_target_final_delta: Dict[str, float] = {}  # key: qid|docid

    for ti, t in enumerate(targets):
        qid, docid = t["qid"], t["docid"]
        orig_enc = encode_clean(tokenizer, t["query"], t["passage"], max_length, device)
        o_enc, c_enc, a_enc, result = build_variant_encodings(
            tokenizer, t["query"], t["passage"], t["attacked_passage"], max_length, device
        )
        if result.status != "ok":
            # Should not happen (Stage 02 already filtered), but never silently skip.
            print(f"[03] WARNING: re-alignment failed for {qid}|{docid}: {result.reason}")
            continue

        variant_encs = {
            "original": orig_enc,
            "padded_control": c_enc,
            "attack": a_enc,
        }
        variant_layer_scores: Dict[str, List[float]] = {}
        for variant, enc in variant_encs.items():
            per_layer = score_target_variant_layerwise(
                model, enc, true_id, false_id, aln
            )
            variant_layer_scores[variant] = [s for (s, _t, _f) in per_layer]
            for li, (s, tl, fl) in enumerate(per_layer):
                target_rows.append({
                    "attack_name": spec.attack_name,
                    "token": spec.token,
                    "position": spec.position,
                    "repetitions": spec.repetitions,
                    "qid": qid,
                    "docid": docid,
                    "variant": variant,
                    "layer_index": li,
                    "layer_name": layer_name(li),
                    "score": s,
                    "true_logit": tl,
                    "false_logit": fl,
                })

        # Final-layer sanity: DecoderLens top layer vs normal monoT5 (attack).
        normal_attack = normal_monot5_score(model, a_enc, true_id, false_id)
        dl_attack_final = variant_layer_scores["attack"][n_layers - 1]
        final_layer_diffs.append(abs(dl_attack_final - normal_attack))

        # Final-layer attack vs control delta (for selection modes).
        fl_attack = variant_layer_scores["attack"][n_layers - 1]
        fl_control = variant_layer_scores["padded_control"][n_layers - 1]
        per_target_final_delta[f"{qid}|{docid}"] = fl_attack - fl_control

        if (ti + 1) % 25 == 0:
            print(f"[03] {spec.attack_name}: scored {ti + 1}/{len(targets)} targets")

    # --- Apply selection filter ---------------------------------------------
    def _key(row: Dict) -> str:
        return f"{row['qid']}|{row['docid']}"

    all_keys = list(per_target_final_delta.keys())
    if mode == "all_aligned":
        selected_keys = set(all_keys)
    elif mode == "final_layer_success":
        selected_keys = {k for k in all_keys if per_target_final_delta[k] > min_delta}
    elif mode == "top_delta":
        ranked = sorted(all_keys, key=lambda k: per_target_final_delta[k], reverse=True)
        selected_keys = set(ranked[:max_targets])
    else:
        raise ValueError(f"Unknown selection.mode: {mode}")

    # Cap at max_targets deterministically (first N by original order).
    if len(selected_keys) > max_targets:
        ordered = [k for k in all_keys if k in selected_keys][:max_targets]
        selected_keys = set(ordered)

    target_rows = [r for r in target_rows if _key(r) in selected_keys]

    # --- Write target scores -------------------------------------------------
    scores_df = pd.DataFrame(target_rows)
    scores_csv = out_dirs["scores"] / "layerwise_target_scores.csv"
    scores_df.to_csv(scores_csv, index=False)
    print(f"[03] {spec.attack_name}: wrote {len(scores_df)} target score rows -> {scores_csv}")

    # --- Write candidate clean scores parquet (for used qids only) -----------
    if bool(cfg["runtime"].get("save_candidate_scores", True)):
        cand_rows: List[Dict] = []
        selected_qids = {k.split("|", 1)[0] for k in selected_keys}
        for qid in selected_qids:
            for docid, layer_scores in candidate_scores.get(qid, {}).items():
                for li, s in enumerate(layer_scores):
                    cand_rows.append({
                        "qid": qid, "docid": docid,
                        "layer_index": li, "score": s,
                    })
        cand_df = pd.DataFrame(cand_rows)
        cand_parquet = out_dirs["scores"] / "layerwise_candidate_scores.parquet"
        cand_df.to_parquet(cand_parquet, index=False)
        print(f"[03] {spec.attack_name}: wrote {len(cand_df)} candidate score rows "
              f"-> {cand_parquet}")

    # --- Sanity JSON ---------------------------------------------------------
    tol = float(cfg["sanity"]["tolerance"])
    max_diff = max(final_layer_diffs) if final_layer_diffs else 0.0
    mean_diff = (sum(final_layer_diffs) / len(final_layer_diffs)) if final_layer_diffs else 0.0
    passed = bool(max_diff <= tol)
    sanity = {
        "attack_name": spec.attack_name,
        "max_abs_score_diff": max_diff,
        "mean_abs_score_diff": mean_diff,
        "tolerance": tol,
        "passed": passed,
    }
    with open(out_dirs["sanity"] / "final_layer_equivalence.json", "w", encoding="utf-8") as fh:
        json.dump(sanity, fh, indent=2)
    if not passed:
        print(f"[03] WARNING: {spec.attack_name} final-layer sanity FAILED "
              f"(max_abs_diff={max_diff:.4f} > tol={tol}).")
    else:
        print(f"[03] {spec.attack_name}: sanity passed (max_abs_diff={max_diff:.5f}).")

    return {
        "n_examples_used": len(selected_keys),
        "sanity_passed": passed,
        "sanity_max_abs_diff": max_diff,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run DecoderLens layerwise scoring.")
    p.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "default.yaml"))
    p.add_argument("--attack-name", default=None,
                   help="Single attack to score (default: all discovered).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    device = resolve_device(cfg["model"]["device"])
    model, tokenizer = load_monot5(cfg["model"]["checkpoint"], device)
    true_id, false_id = get_true_false_token_ids(tokenizer)
    base_dir = PROJECT_ROOT / cfg["outputs"]["base_dir"]

    attacks = discover_attacks(cfg["attacks"])
    if args.attack_name:
        attacks = [a for a in attacks if a.attack_name == args.attack_name]
        if not attacks:
            raise SystemExit(f"Attack '{args.attack_name}' not found.")

    for spec in attacks:
        score_attack(spec, cfg, model, tokenizer, true_id, false_id, device, base_dir)


if __name__ == "__main__":
    main()
