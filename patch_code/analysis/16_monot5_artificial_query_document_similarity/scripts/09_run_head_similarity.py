"""
scripts/09_run_head_similarity.py
==================================
Stage 09 — head-level query/document similarity (model stage) at the
canonical important heads (exp16lib.heads): 18 encoder self-attention heads
(pre-o_proj 64-d head output, pooled query vs pooled document) and 31 decoder
cross-attention heads (first decoder step; query-sourced vs document-sourced
64-d contribution).

Populations (same manifests, encodings and masks as stages 02/03):
  * balanced DL19 qrel sample, clean prompts (qrel 2/3 and 0);
  * ALL attack examples x ALL attacks, the ATTACKED input (injected tokens in
    the document span). Success is not filtered here (split in stage 10).

One wide row per example: meta + enc_<label> (18) + dec_<label> (31).
Resume: qrel = one unit; attacks = one unit per attack (manifest sha guard).
Outputs: 09_heads/qrel_heads.csv.gz, 09_heads/attack_heads.csv.gz
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
from exp16lib.heads import decoder_heads, encoder_heads, run_head_sequences  # noqa: E402
from exp16lib.inputs import encode_attack_and_control, encode_clean  # noqa: E402
from exp16lib.run_utils import load_manifest, load_model, now, stage_argparser, unit_is_done, write_status  # noqa: E402

DT = {"qid": str, "docid": str, "pair_id": str, "attack_token": str, "attack_name": str}
ATTACK_META = ["attack_id", "attack_name", "attack_token", "attack_position", "repetitions", "pair_id", "qid",
               "docid", "score_attack", "score_control", "delta_score"]
QREL_META = ["qid", "docid", "pair_id", "qrel_grade", "relevance_group"]


def wide(meta, E, D, enc_h, dec_h):
    df = pd.DataFrame(meta)
    for j, h in enumerate(enc_h):
        df[f"enc_{h.label}"] = E[:, j]
    for j, h in enumerate(dec_h):
        df[f"dec_{h.label}"] = D[:, j]
    return df


def main():
    p = stage_argparser("Exp 16 stage 09: head-level query/document similarity")
    p.add_argument("--attack-start", type=int, default=0)
    p.add_argument("--attack-end", type=int, default=None)
    args = p.parse_args()
    cfg = load_config(args.config)
    out = output_dir(cfg)
    stage = out / "09_heads"
    qrel, q_sha = load_manifest(out, "qrel")
    atk, a_sha = load_manifest(out, "attacks")
    enc_h, dec_h = encoder_heads(), decoder_heads()
    rtol = float(cfg["validation"]["decoder_decomposition_rtol"])
    bs = int(cfg["runtime"]["batch_size"])
    max_len = int(cfg["model"]["max_length"])

    order = list(dict.fromkeys(r["attack_name"] for r in atk))[args.attack_start:args.attack_end]
    q_done = unit_is_done(stage / "qrel", ["rows.csv.gz"], q_sha, args.force)
    pending = [a for a in order if not unit_is_done(stage / "per_attack" / a, ["rows.csv.gz"], a_sha, args.force)]
    print(f"[09] heads: {len(enc_h)} encoder, {len(dec_h)} decoder cross-attn; qrel {'done' if q_done else 'to run'}; "
          f"attacks: {len(order)} selected, {len(pending)} to run")
    if not q_done or pending:
        model, tok, true_id, false_id, device = load_model(cfg)
        pad = tok.pad_token_id
        if not q_done:
            unit = stage / "qrel"
            write_status(unit, {"status": "running", "started": now()})
            seqs = [encode_clean(tok, r["query"], r["passage"], max_len) for r in qrel]
            E, D, err = run_head_sequences(model, seqs, pad, device, bs, enc_h, dec_h, rtol)
            wide([{k: r[k] for k in QREL_META} for r in qrel], E, D, enc_h, dec_h).to_csv(unit / "rows.csv.gz", index=False)
            write_status(unit, {"status": "success", "finished": now(), "manifest_sha256": q_sha, "n_docs": len(qrel),
                                "decoder_decomposition_max_rel": err, "device": str(device)})
            print(f"[09] qrel: {len(qrel)} docs, decoder rel err {err:.1e}", flush=True)
        t_stage = time.time()
        for k, a in enumerate(pending):
            t0 = time.time()
            unit = stage / "per_attack" / a
            write_status(unit, {"status": "running", "started": now()})
            ra = [r for r in atk if r["attack_name"] == a]
            seqs = [encode_attack_and_control(tok, r["query"], r["passage"], r["attacked_passage"], max_len)[0] for r in ra]
            E, D, err = run_head_sequences(model, seqs, pad, device, bs, enc_h, dec_h, rtol)
            wide([{m: r[m] for m in ATTACK_META} for r in ra], E, D, enc_h, dec_h).to_csv(unit / "rows.csv.gz", index=False)
            write_status(unit, {"status": "success", "finished": now(), "manifest_sha256": a_sha, "n_examples": len(ra),
                                "decoder_decomposition_max_rel": err, "seconds": time.time() - t0, "device": str(device)})
            el = time.time() - t_stage
            print(f"[09] {a:24s} {len(ra)} ex, rel err {err:.1e}, {time.time() - t0:.1f}s ({k + 1}/{len(pending)}, "
                  f"ETA {el / (k + 1) * (len(pending) - k - 1) / 60:.1f} min)", flush=True)

    all_attacks = list(dict.fromkeys(r["attack_name"] for r in atk))
    missing = [a for a in all_attacks if not unit_is_done(stage / "per_attack" / a, ["rows.csv.gz"], a_sha, False)]
    if missing or not unit_is_done(stage / "qrel", ["rows.csv.gz"], q_sha, False):
        print(f"[09] {len(missing)} attacks (or qrel) still missing; combined tables not written yet")
        return
    pd.read_csv(stage / "qrel" / "rows.csv.gz", dtype=DT).to_csv(stage / "qrel_heads.csv.gz", index=False)
    df = pd.concat([pd.read_csv(stage / "per_attack" / a / "rows.csv.gz", dtype=DT) for a in all_attacks],
                   ignore_index=True)
    if not np.allclose(df.delta_score, df.score_attack - df.score_control, rtol=0, atol=1e-9):
        raise RuntimeError("delta_score != score_attack - score_control")
    df.to_csv(stage / "attack_heads.csv.gz", index=False)
    write_status(stage, {"status": "success", "finished": now(), "qrel_manifest_sha256": q_sha,
                         "attack_manifest_sha256": a_sha, "n_attack_examples": len(df),
                         "encoder_heads": [h.label for h in enc_h], "decoder_heads": [h.label for h in dec_h]})
    print(f"[09] combined -> {stage}")


if __name__ == "__main__":
    main()
