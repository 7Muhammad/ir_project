"""
scripts/11_run_decoder_probe_similarity.py
===========================================
Stage 11 — decoder residual-stream query/document similarity (model stage):
cos(decoder state | cross-attention restricted to query-text positions,
    decoder state | cross-attention restricted to document positions)
at 37 decoder checkpoints (exp16lib.decoder_probe).

Populations (same manifests / encodings / masks as stages 02/03): the balanced
DL19 qrel sample (clean prompts) and ALL attack examples x ALL attacks
(ATTACKED input; injected tokens in the document span). Success split in stage 12.

One wide row per example: meta + one column per checkpoint.
Resume: qrel = one unit; attacks = one unit per attack (manifest sha guard).
Outputs: 11_decoder_probe/qrel_decoder_probe.csv.gz, 11_decoder_probe/attack_decoder_probe.csv.gz
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
from exp16lib.decoder_probe import DECODER_PROBE_CHECKPOINTS, run_decoder_probe  # noqa: E402
from exp16lib.inputs import encode_attack_and_control, encode_clean  # noqa: E402
from exp16lib.run_utils import load_manifest, load_model, now, stage_argparser, unit_is_done, write_status  # noqa: E402

DT = {"qid": str, "docid": str, "pair_id": str, "attack_token": str, "attack_name": str}
ATTACK_META = ["attack_id", "attack_name", "attack_token", "attack_position", "repetitions", "pair_id", "qid",
               "docid", "score_attack", "score_control", "delta_score"]
QREL_META = ["qid", "docid", "pair_id", "qrel_grade", "relevance_group"]


def wide(meta, S):
    df = pd.DataFrame(meta)
    for j, c in enumerate(DECODER_PROBE_CHECKPOINTS):
        df[c] = S[:, j]
    return df


def check_trivial(S, where):
    """dec_embedding and D00_post_self_attn precede any cross-attention -> must be exactly identical (cos 1)."""
    dev = float(np.abs(S[:, :2] - 1.0).max())
    if dev > 1e-9:
        raise AssertionError(f"{where}: pre-cross-attention checkpoints differ between probe runs (max |1-cos| {dev})")
    return dev


def main():
    p = stage_argparser("Exp 16 stage 11: decoder query-only vs doc-only probe similarity")
    p.add_argument("--attack-start", type=int, default=0)
    p.add_argument("--attack-end", type=int, default=None)
    args = p.parse_args()
    cfg = load_config(args.config)
    out = output_dir(cfg)
    stage = out / "11_decoder_probe"
    qrel, q_sha = load_manifest(out, "qrel")
    atk, a_sha = load_manifest(out, "attacks")
    bs = int(cfg["runtime"]["batch_size"])
    max_len = int(cfg["model"]["max_length"])

    order = list(dict.fromkeys(r["attack_name"] for r in atk))[args.attack_start:args.attack_end]
    q_done = unit_is_done(stage / "qrel", ["rows.csv.gz"], q_sha, args.force)
    pending = [a for a in order if not unit_is_done(stage / "per_attack" / a, ["rows.csv.gz"], a_sha, args.force)]
    print(f"[11] qrel {'done' if q_done else 'to run'}; attacks: {len(order)} selected, {len(pending)} to run")
    if not q_done or pending:
        model, tok, _, _, device = load_model(cfg)
        pad = tok.pad_token_id
        if not q_done:
            unit = stage / "qrel"
            write_status(unit, {"status": "running", "started": now()})
            S = run_decoder_probe(model, [encode_clean(tok, r["query"], r["passage"], max_len) for r in qrel], pad, device, bs)
            dev = check_trivial(S, "qrel")
            wide([{k: r[k] for k in QREL_META} for r in qrel], S).to_csv(unit / "rows.csv.gz", index=False)
            write_status(unit, {"status": "success", "finished": now(), "manifest_sha256": q_sha, "n_docs": len(qrel),
                                "pre_cross_attn_max_dev": dev, "device": str(device)})
            print(f"[11] qrel: {len(qrel)} docs", flush=True)
        t_stage = time.time()
        for k, a in enumerate(pending):
            t0 = time.time()
            unit = stage / "per_attack" / a
            write_status(unit, {"status": "running", "started": now()})
            ra = [r for r in atk if r["attack_name"] == a]
            seqs = [encode_attack_and_control(tok, r["query"], r["passage"], r["attacked_passage"], max_len)[0] for r in ra]
            S = run_decoder_probe(model, seqs, pad, device, bs)
            dev = check_trivial(S, a)
            wide([{m: r[m] for m in ATTACK_META} for r in ra], S).to_csv(unit / "rows.csv.gz", index=False)
            write_status(unit, {"status": "success", "finished": now(), "manifest_sha256": a_sha, "n_examples": len(ra),
                                "pre_cross_attn_max_dev": dev, "seconds": time.time() - t0, "device": str(device)})
            el = time.time() - t_stage
            print(f"[11] {a:24s} {len(ra)} ex, {time.time() - t0:.1f}s ({k + 1}/{len(pending)}, "
                  f"ETA {el / (k + 1) * (len(pending) - k - 1) / 60:.1f} min)", flush=True)

    all_attacks = list(dict.fromkeys(r["attack_name"] for r in atk))
    missing = [a for a in all_attacks if not unit_is_done(stage / "per_attack" / a, ["rows.csv.gz"], a_sha, False)]
    if missing or not unit_is_done(stage / "qrel", ["rows.csv.gz"], q_sha, False):
        print(f"[11] {len(missing)} attacks (or qrel) still missing; combined tables not written yet")
        return
    pd.read_csv(stage / "qrel" / "rows.csv.gz", dtype=DT).to_csv(stage / "qrel_decoder_probe.csv.gz", index=False)
    df = pd.concat([pd.read_csv(stage / "per_attack" / a / "rows.csv.gz", dtype=DT) for a in all_attacks],
                   ignore_index=True)
    df.to_csv(stage / "attack_decoder_probe.csv.gz", index=False)
    write_status(stage, {"status": "success", "finished": now(), "qrel_manifest_sha256": q_sha,
                         "attack_manifest_sha256": a_sha, "n_attack_examples": len(df)})
    print(f"[11] combined -> {stage}")


if __name__ == "__main__":
    main()
