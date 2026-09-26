"""
scripts/13_run_late_encoder_heads.py
=====================================
Stage 13 — head-level query/document similarity at ALL 36 encoder
self-attention heads of layers 9-11 (L9H0..L11H11), not only the canonical
important ones. Same definition as stage 09 (exp16lib.heads.EncoderHeadCapture):
the 64-d head output BEFORE SelfAttention.o, mean-pooled over query-text vs
document positions, cosine. Encoder-only forward (no decoder needed).

Populations = stage 09: balanced DL19 qrel sample (clean prompts) and ALL
attack examples x ALL attacks, ATTACKED input (injected tokens in the document
span). Success split happens in stage 14.

Cross-check: the 16 heads shared with stage 09's canonical set must reproduce
09_heads values (encoder-only vs full forward, different batch composition only).

Resume: qrel = one unit; attacks = one unit per attack (manifest sha guard).
Outputs: 13_late_encoder_heads/{qrel,attack}_late_encoder_heads.csv.gz
"""

from __future__ import annotations

import pathlib
import sys
import time

import numpy as np
import pandas as pd

EXP_DIR = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EXP_DIR))
import exp16lib  # noqa: E402,F401

from exp16lib.config import load_config, output_dir  # noqa: E402
from exp16lib.heads import all_encoder_heads, run_encoder_head_sequences  # noqa: E402
from exp16lib.inputs import encode_attack_and_control, encode_clean  # noqa: E402
from exp16lib.run_utils import (is_already_successful, load_manifest, load_model, now, stage_argparser,  # noqa: E402
                                unit_is_done, write_status)

LAYERS = [9, 10, 11]
DT = {"qid": str, "docid": str, "pair_id": str, "attack_token": str, "attack_name": str}
ATTACK_META = ["attack_id", "attack_name", "attack_token", "attack_position", "repetitions", "pair_id", "qid",
               "docid", "score_attack", "score_control", "delta_score"]
QREL_META = ["qid", "docid", "pair_id", "qrel_grade", "relevance_group"]


def wide(meta, E, heads):
    df = pd.DataFrame(meta)
    for j, h in enumerate(heads):
        df[f"enc_{h.label}"] = E[:, j]
    return df


def cross_check(new: pd.DataFrame, old_path: pathlib.Path, keys, atol: float):
    """Shared heads must match the stage-09 cache row for row."""
    if not old_path.exists():
        return None
    old = pd.read_csv(old_path, dtype=DT)
    shared = [c for c in new.columns if c.startswith("enc_") and c in old.columns]
    m = new[keys + shared].merge(old[keys + shared], on=keys, suffixes=("", "_09"), validate="one_to_one")
    if len(m) != len(new):
        raise RuntimeError(f"cross-check: {len(m)} of {len(new)} rows matched stage 09")
    worst = max(float((m[c] - m[f"{c}_09"]).abs().max()) for c in shared)
    if worst > atol:
        raise AssertionError(f"shared heads differ from stage 09 by {worst} > {atol}")
    return {"n_shared_heads": len(shared), "max_abs_diff_vs_stage09": worst}


def main():
    p = stage_argparser("Exp 16 stage 13: all encoder heads of layers 9-11")
    p.add_argument("--attack-start", type=int, default=0)
    p.add_argument("--attack-end", type=int, default=None)
    args = p.parse_args()
    cfg = load_config(args.config)
    out = output_dir(cfg)
    stage = out / "13_late_encoder_heads"
    heads = all_encoder_heads(LAYERS)
    assert len(heads) == 36
    qrel, q_sha = load_manifest(out, "qrel")
    atk, a_sha = load_manifest(out, "attacks")
    bs = int(cfg["runtime"]["batch_size"])
    max_len = int(cfg["model"]["max_length"])

    order = list(dict.fromkeys(r["attack_name"] for r in atk))[args.attack_start:args.attack_end]
    q_done = unit_is_done(stage / "qrel", ["rows.csv.gz"], q_sha, args.force)
    pending = [a for a in order if not unit_is_done(stage / "per_attack" / a, ["rows.csv.gz"], a_sha, args.force)]
    print(f"[13] {len(heads)} heads (layers {LAYERS}); qrel {'done' if q_done else 'to run'}; "
          f"attacks: {len(order)} selected, {len(pending)} to run")
    if not q_done or pending:
        model, tok, _, _, device = load_model(cfg)
        pad = tok.pad_token_id
        if not q_done:
            unit = stage / "qrel"
            write_status(unit, {"status": "running", "started": now()})
            E = run_encoder_head_sequences(model, [encode_clean(tok, r["query"], r["passage"], max_len) for r in qrel],
                                           pad, device, bs, heads)
            wide([{k: r[k] for k in QREL_META} for r in qrel], E, heads).to_csv(unit / "rows.csv.gz", index=False)
            write_status(unit, {"status": "success", "finished": now(), "manifest_sha256": q_sha, "n_docs": len(qrel),
                                "heads": [h.label for h in heads], "device": str(device)})
            print(f"[13] qrel: {len(qrel)} docs", flush=True)
        t_stage = time.time()
        for k, a in enumerate(pending):
            t0 = time.time()
            unit = stage / "per_attack" / a
            write_status(unit, {"status": "running", "started": now()})
            ra = [r for r in atk if r["attack_name"] == a]
            seqs = [encode_attack_and_control(tok, r["query"], r["passage"], r["attacked_passage"], max_len)[0] for r in ra]
            E = run_encoder_head_sequences(model, seqs, pad, device, bs, heads)
            wide([{m: r[m] for m in ATTACK_META} for r in ra], E, heads).to_csv(unit / "rows.csv.gz", index=False)
            write_status(unit, {"status": "success", "finished": now(), "manifest_sha256": a_sha, "n_examples": len(ra),
                                "heads": [h.label for h in heads], "seconds": time.time() - t0, "device": str(device)})
            el = time.time() - t_stage
            print(f"[13] {a:24s} {len(ra)} ex, {time.time() - t0:.1f}s ({k + 1}/{len(pending)}, "
                  f"ETA {el / (k + 1) * (len(pending) - k - 1) / 60:.1f} min)", flush=True)

    all_attacks = list(dict.fromkeys(r["attack_name"] for r in atk))
    missing = [a for a in all_attacks if not unit_is_done(stage / "per_attack" / a, ["rows.csv.gz"], a_sha, False)]
    if missing or not unit_is_done(stage / "qrel", ["rows.csv.gz"], q_sha, False):
        print(f"[13] {len(missing)} attacks (or qrel) still missing; combined tables not written yet")
        return
    q = pd.read_csv(stage / "qrel" / "rows.csv.gz", dtype=DT)
    a = pd.concat([pd.read_csv(stage / "per_attack" / x / "rows.csv.gz", dtype=DT) for x in all_attacks], ignore_index=True)
    q.to_csv(stage / "qrel_late_encoder_heads.csv.gz", index=False)
    a.to_csv(stage / "attack_late_encoder_heads.csv.gz", index=False)
    checks = {}
    if is_already_successful(out / "09_heads", ["qrel_heads.csv.gz", "attack_heads.csv.gz"]):
        atol = float(cfg["validation"].get("head_crosscheck_atol", 1e-4))
        checks["qrel"] = cross_check(q, out / "09_heads" / "qrel_heads.csv.gz", ["pair_id", "relevance_group"], atol)
        checks["attack"] = cross_check(a, out / "09_heads" / "attack_heads.csv.gz", ["attack_name", "pair_id"], atol)
        print(f"[13] cross-check vs stage 09: {checks}")
    write_status(stage, {"status": "success", "finished": now(), "qrel_manifest_sha256": q_sha,
                         "attack_manifest_sha256": a_sha, "n_attack_examples": len(a),
                         "heads": [h.label for h in heads], "crosscheck_vs_stage09": checks})
    print(f"[13] combined -> {stage}")


if __name__ == "__main__":
    main()
