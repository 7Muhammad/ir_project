"""
scripts/07_run_decoder_message_similarity.py
=============================================
Stage 07 — decoder-side query/document message similarity (model stage).

For every decoder layer: cos(m_query, m_doc) of the cross-attention output
split by source span (exp16lib.decoder), plus cross-attention mass and message
norms, for
  * the balanced DL19 qrel sample (clean prompts, qrel 2/3 and qrel 0);
  * ALL attack examples x ALL attacks, attacked (prefix a_) AND matched padded
    control (prefix c_), no success filter.
Same manifests, encodings and span masks as stages 02/03 (injected tokens in
the attacked document span; masked control slots excluded).

Every batch checks m_query + m_doc + m_rest == the cross-attention output
(relative error <= validation.decoder_decomposition_rtol). The monoT5 score of the same forward is compared
with the cached Exp 01 control/attack scores for EVERY attack example (free
by-product; fails loudly above validation.fresh_vs_cached_score_atol).

Resume: qrel = one unit; attacks = one unit per attack (manifest sha guard).
Outputs: 07_decoder/qrel_decoder.csv.gz, 07_decoder/attack_decoder.csv.gz
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
from exp16lib.decoder import DECODER_CHECKPOINTS, FIELDS, run_decoder_sequences  # noqa: E402
from exp16lib.inputs import encode_attack_and_control, encode_clean  # noqa: E402
from exp16lib.run_utils import load_manifest, load_model, now, stage_argparser, unit_is_done, write_status  # noqa: E402

DT = {"qid": str, "docid": str, "pair_id": str, "attack_token": str, "attack_name": str}
ATTACK_META = ["attack_id", "attack_name", "attack_token", "attack_position", "repetitions", "pair_id", "qid",
               "docid", "score_attack", "score_control", "delta_score"]
QREL_META = ["qid", "docid", "pair_id", "qrel_grade", "relevance_group"]


def long_decoder(meta, values):
    n, L = len(meta), len(DECODER_CHECKPOINTS)
    m = pd.DataFrame(meta).loc[np.repeat(np.arange(n), L)].reset_index(drop=True)
    m["decoder_layer"] = np.tile(np.arange(L), n)
    m["checkpoint_name"] = np.tile(DECODER_CHECKPOINTS, n)
    for k, v in values.items():
        if v.shape != (n, L):
            raise ValueError(f"{k}: {v.shape}")
        m[k] = v.reshape(-1)
    return m


def main():
    p = stage_argparser("Exp 16 stage 07: decoder cross-attention message similarity")
    p.add_argument("--attack-start", type=int, default=0)
    p.add_argument("--attack-end", type=int, default=None)
    args = p.parse_args()
    cfg = load_config(args.config)
    out = output_dir(cfg)
    stage = out / "07_decoder"
    qrel, q_sha = load_manifest(out, "qrel")
    atk, a_sha = load_manifest(out, "attacks")
    val = cfg["validation"]
    dec_rtol = float(val["decoder_decomposition_rtol"])
    score_atol = float(val["fresh_vs_cached_score_atol"])
    bs = int(cfg["runtime"]["batch_size"])
    max_len = int(cfg["model"]["max_length"])

    order = list(dict.fromkeys(r["attack_name"] for r in atk))[args.attack_start:args.attack_end]
    q_done = unit_is_done(stage / "qrel", ["rows.csv.gz"], q_sha, args.force)
    pending = [a for a in order if not unit_is_done(stage / "per_attack" / a, ["rows.csv.gz"], a_sha, args.force)]
    print(f"[07] qrel {'done' if q_done else 'to run'}; attacks: {len(order)} selected, {len(pending)} to run")
    if not q_done or pending:
        model, tok, true_id, false_id, device = load_model(cfg)
        pad = tok.pad_token_id
        if not q_done:
            unit = stage / "qrel"
            write_status(unit, {"status": "running", "started": now()})
            seqs = [encode_clean(tok, r["query"], r["passage"], max_len) for r in qrel]
            res, sc, err = run_decoder_sequences(model, seqs, pad, device, bs, true_id, false_id, dec_rtol)
            meta = [{k: r[k] for k in QREL_META} | {"monot5_score": float(s)} for r, s in zip(qrel, sc)]
            long_decoder(meta, res).to_csv(unit / "rows.csv.gz", index=False)
            write_status(unit, {"status": "success", "finished": now(), "manifest_sha256": q_sha, "n_docs": len(qrel),
                                "decomposition_max_rel": err, "device": str(device)})
            print(f"[07] qrel: {len(qrel)} docs, relative decomposition err {err:.2e}", flush=True)
        t_stage = time.time()
        for k, a in enumerate(pending):
            t0 = time.time()
            unit = stage / "per_attack" / a
            write_status(unit, {"status": "running", "started": now()})
            ra = [r for r in atk if r["attack_name"] == a]
            a_seqs, c_seqs = [], []
            for r in ra:
                s_a, s_c, _ = encode_attack_and_control(tok, r["query"], r["passage"], r["attacked_passage"], max_len)
                a_seqs.append(s_a)
                c_seqs.append(s_c)
            res, sc, err = run_decoder_sequences(model, a_seqs + c_seqs, pad, device, bs, true_id, false_id, dec_rtol)
            n = len(ra)
            cached = np.array([r["score_attack"] for r in ra] + [r["score_control"] for r in ra])
            sdiff = float(np.abs(sc - cached).max())
            if sdiff > score_atol:
                raise AssertionError(f"{a}: fresh vs cached score diff {sdiff} > {score_atol}")
            vals = {f"a_{f}": res[f][:n] for f in FIELDS} | {f"c_{f}": res[f][n:] for f in FIELDS}
            long_decoder([{m: r[m] for m in ATTACK_META} for r in ra], vals).to_csv(unit / "rows.csv.gz", index=False)
            write_status(unit, {"status": "success", "finished": now(), "manifest_sha256": a_sha, "n_examples": n,
                                "decomposition_max_rel": err, "fresh_vs_cached_score_max_abs": sdiff,
                                "seconds": time.time() - t0, "device": str(device)})
            el = time.time() - t_stage
            print(f"[07] {a:24s} {n} ex, rel err {err:.1e}, score diff {sdiff:.1e}, {time.time() - t0:.1f}s "
                  f"({k + 1}/{len(pending)}, ETA {el / (k + 1) * (len(pending) - k - 1) / 60:.1f} min)", flush=True)

    all_attacks = list(dict.fromkeys(r["attack_name"] for r in atk))
    missing = [a for a in all_attacks if not unit_is_done(stage / "per_attack" / a, ["rows.csv.gz"], a_sha, False)]
    if missing or not unit_is_done(stage / "qrel", ["rows.csv.gz"], q_sha, False):
        print(f"[07] {len(missing)} attacks (or qrel) still missing; combined tables not written yet")
        return
    pd.read_csv(stage / "qrel" / "rows.csv.gz", dtype=DT).to_csv(stage / "qrel_decoder.csv.gz", index=False)
    df = pd.concat([pd.read_csv(stage / "per_attack" / a / "rows.csv.gz", dtype=DT) for a in all_attacks],
                   ignore_index=True)
    df.to_csv(stage / "attack_decoder.csv.gz", index=False)
    write_status(stage, {"status": "success", "finished": now(), "qrel_manifest_sha256": q_sha,
                         "attack_manifest_sha256": a_sha, "n_attacks": len(all_attacks), "n_attack_rows": len(df)})
    print(f"[07] combined -> {stage}")


if __name__ == "__main__":
    main()
