"""
scripts/22_extract_token_sample.py
===================================
Stage 22 — per-token head outputs (all 144 encoder heads, pre-o_proj) for a small fixed
exploration sample; format and loader in exp16lib/token_sample.py.

Sample (config token_sample): from the stage-17 paired manifest, up to `docs_per_group`
qrel 2/3 and qrel 0 base pairs per query (seed 42); for each: the clean input and, for
each attack of the 12-attack subset in which the pair is aligned, the padded control and
the attacked input (exp16lib.inputs.encode_clean / encode_attack_and_control — the same
encodings as stages 18 and 00). No success filter (success is a column).

One forward per length-sorted batch with EncoderHeadCapture(keep_inputs=True) attached to
exp16lib.engine.forward_batch(with_score=True).

Checks (fail loudly):
  * the capture's own 144 head cosines == stage 18 (all_layers) values for the same control/attack
    sequences and == the diagnostic clean values when present (atol validation.head_crosscheck_atol);
    scores == stage 18 (atol validation.fresh_vs_cached_score_atol)
  * the same cosines recomputed from the STORED fp16 tokens agree within 5e-3 (fp16 storage error)

Outputs: 22_token_sample/{heads.npy, tokens.npz, sequences.csv, checks.json, status.json}
"""

from __future__ import annotations

import json
import pathlib
import sys
import time

import numpy as np
import pandas as pd
import torch

EXP_DIR = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EXP_DIR))
import exp16lib  # noqa: E402,F401

from exp16lib import paired as P  # noqa: E402
from exp16lib import token_sample as TS  # noqa: E402
from exp16lib.config import load_config, output_dir  # noqa: E402
from exp16lib.engine import forward_batch  # noqa: E402
from exp16lib.heads import EncoderHeadCapture, all_encoder_heads  # noqa: E402
from exp16lib.inputs import collate, encode_clean  # noqa: E402
from exp16lib.run_utils import (is_already_successful, load_model, load_status, now, read_jsonl,  # noqa: E402
                                stage_argparser, write_status)

STAGE = "22_token_sample"


def main():
    args = stage_argparser("Exp 16 stage 22: per-token head outputs for an exploration sample").parse_args()
    cfg = load_config(args.config)
    P.validate_cfg(cfg)
    tc = cfg["token_sample"]
    out = output_dir(cfg)
    st = out / STAGE
    if not args.force and is_already_successful(st, ["heads.npy", "tokens.npz", "sequences.csv"]):
        print(f"[22] already complete in {st} (use --force)")
        return
    st.mkdir(parents=True, exist_ok=True)
    write_status(st, {"status": "running", "started": now()})
    t0 = time.time()
    mdir = out / P.MANIFEST_DIR
    attacks = tc.get("attacks") or TS.DEFAULT_ATTACKS
    grid = list(load_status(mdir)["sha256"]["attacks"])
    missing = [a for a in attacks if a not in grid]
    if missing:
        raise RuntimeError(f"attacks not in the paired manifest: {missing}")
    base = read_jsonl(mdir / "base_pairs.jsonl.gz")
    pairs = TS.sample_pairs(base, int(tc["docs_per_group"]), int(tc["seed"]))
    want = {p["pair_id"] for p in pairs}

    model, tok, tid, fid, dev = load_model(cfg)
    max_len = int(cfg["model"]["max_length"])
    seqs, meta = [], []

    def add(seq, inserted, m):
        seqs.append(seq)
        ins = np.zeros(seq.seq_len, np.int8)
        ins[list(inserted)] = 1
        meta.append({**m, "n_tokens": seq.seq_len, "_ins": ins})

    for p in pairs:
        add(encode_clean(tok, p["query"], p["passage"], max_len), [],
            {"kind": "clean", "attack_name": "", **{k: p[k] for k in ("pair_id", "qid", "docid", "qrel_grade", "relevance_group")}})
    n_missing_inst = 0
    for a in attacks:
        recs = {r["pair_id"]: r for r in read_jsonl(mdir / "attacks" / f"{a}.jsonl.gz") if r["pair_id"] in want}
        n_missing_inst += len(want) - len(recs)
        for p in pairs:
            r = recs.get(p["pair_id"])
            if r is None:                                            # alignment failure (logged in stage 17)
                continue
            atk, ctl, info = P.encode_instance(tok, p["query"], p["passage"], r["attacked_passage"], max_len)
            m = {"attack_name": a, "attack_token": r["attack_token"], "attack_position": r["attack_position"],
                 "repetitions": r["repetitions"], **{k: p[k] for k in ("pair_id", "qid", "docid", "qrel_grade", "relevance_group")}}
            add(ctl, info["inserted_positions"], {"kind": "control", **m})
            add(atk, info["inserted_positions"], {"kind": "attack", **m})

    n = len(seqs)
    lens = np.array([s.seq_len for s in seqs])
    offs = np.concatenate([[0], np.cumsum(lens)])
    total = int(offs[-1])
    print(f"[22] {len(pairs)} pairs ({sum(p['relevance_group'] == 'relevant' for p in pairs)} rel), {len(attacks)} attacks, "
          f"{n} sequences, {total} tokens -> {total * 144 * 64 * 2 / 2**30:.1f} GiB fp16", flush=True)
    heads_mm = np.lib.format.open_memmap(st / "heads.npy.tmp", mode="w+", dtype=np.float16,
                                         shape=(total, TS.N_LAYERS, TS.N_HEADS_PER_LAYER, TS.D_KV))
    hs = all_encoder_heads(range(12))
    labels = [h.label for h in hs]
    cos_cap = np.full((n, 144), np.nan)
    scores = np.full(n, np.nan)
    bs = int(tc["batch_size"])
    order = sorted(range(n), key=lambda i: lens[i])
    for start in range(0, n, bs):
        idx = order[start:start + bs]
        batch = collate([seqs[i] for i in idx], tok.pad_token_id, dev)
        with torch.inference_mode(), EncoderHeadCapture(model.encoder, hs, keep_inputs=True) as ec:
            ec.set_masks(batch["query_mask"], batch["doc_mask"])
            _, sc, _ = forward_batch(model, batch, True, tid, fid)
            X = torch.stack([ec.inputs[L] for L in range(12)], 2)    # [B, S, 12, 768]  pre-o_proj
            C = torch.stack([ec.cos[l] for l in labels], 1)
        X = X.reshape(X.shape[0], X.shape[1], 12, 12, 64).to(torch.float16).cpu().numpy()
        for j, i in enumerate(idx):
            heads_mm[offs[i]:offs[i + 1]] = X[j, :lens[i]]
        cos_cap[idx], scores[idx] = C.cpu().numpy(), sc.cpu().numpy()
        if (start // bs) % 20 == 0:
            print(f"[22] {min(start + bs, n)}/{n} sequences, {time.time() - t0:.0f}s", flush=True)
    heads_mm.flush()
    del heads_mm
    (st / "heads.npy.tmp").replace(st / "heads.npy")

    tokens = {"input_ids": np.concatenate([s.input_ids for s in seqs]).astype(np.int32),
              "attention_mask": np.concatenate([s.attention_mask for s in seqs]).astype(np.int8),
              "query_mask": np.concatenate([s.query_mask for s in seqs]).astype(np.int8),
              "doc_mask": np.concatenate([s.doc_mask for s in seqs]).astype(np.int8),
              "inserted_mask": np.concatenate([m.pop("_ins") for m in meta]).astype(np.int8)}
    np.savez(st / "tokens.npz", **tokens)
    df = pd.DataFrame(meta)
    df.insert(0, "seq_id", np.arange(n))
    df.insert(1, "token_offset", offs[:-1])
    df["score"] = scores
    # control/attack pairing -> delta_score, successful (per instance)
    inst = df[df.kind != "clean"].pivot_table(index=["attack_name", "pair_id"], columns="kind", values="score")
    inst["delta_score"] = inst["attack"] - inst["control"]
    df = df.merge(inst["delta_score"].reset_index(), on=["attack_name", "pair_id"], how="left")
    df["successful"] = np.where(df.kind == "clean", None, df.delta_score > 0)      # clean rows: no pair -> empty
    df = pd.concat([df, pd.DataFrame(cos_cap, columns=[f"cos_{l}" for l in labels])], axis=1)
    df.to_csv(st / "sequences.csv", index=False, float_format="%.9g")

    # ---- checks ---------------------------------------------------------------------------------------------
    val = cfg["validation"]
    atol = float(val["head_crosscheck_atol"])
    checks = {"n_pairs": len(pairs), "n_sequences": n, "n_tokens": total,
              "n_instances_missing_alignment": n_missing_inst,
              "by_kind": df.kind.value_counts().to_dict(), "by_group": df.drop_duplicates("pair_id").relevance_group.value_counts().to_dict()}
    P.configure("all_layers")
    f18 = out / P.FORWARD_DIR
    rows = []
    for a in attacks:
        f = f18 / "per_attack" / a / "rows.csv.gz"
        if f.exists():
            rows.append(pd.read_csv(f, dtype=P.DT, usecols=["attack_name", "pair_id", "score_control", "score_attack"]
                                    + P.head_cols("control") + P.head_cols("attack")))
    if rows:
        r = pd.concat(rows)
        worst_c, worst_s, nn = 0.0, 0.0, 0
        for kind in ("control", "attack"):
            m = df[df.kind == kind].merge(r, on=["attack_name", "pair_id"], validate="one_to_one")
            nn += len(m)
            worst_c = max(worst_c, float(np.abs(m[[f"cos_{l}" for l in labels]].values - m[P.head_cols(kind)].values).max()))
            worst_s = max(worst_s, float(np.abs(m.score - m[f"score_{kind}"]).max()))
        checks["vs_stage18_all_layers"] = {"n_sequences": nn, "max_abs_diff_cos": worst_c, "max_abs_diff_score": worst_s}
        if worst_c > atol or worst_s > float(val["fresh_vs_cached_score_atol"]):
            raise AssertionError(f"token-sample capture differs from stage 18: {checks['vs_stage18_all_layers']}")
    cl = out / "diag_paired_all_layers" / "clean_qrel_144_heads.csv.gz"
    if cl.exists():
        c = pd.read_csv(cl, dtype={"pair_id": str})
        m = df[df.kind == "clean"].merge(c, on="pair_id")
        if len(m):
            d = float(np.abs(m[[f"cos_{l}" for l in labels]].values - m[[f"clean_head_{l}" for l in labels]].values).max())
            checks["vs_diag_clean"] = {"n_sequences": len(m), "max_abs_diff_cos": d}
            if d > atol:
                raise AssertionError(f"clean capture differs from the diagnostic: {d}")
    ts = TS.TokenSample.__new__(TS.TokenSample)                    # recompute from the STORED fp16 tokens
    ts.dir, ts.meta = st, df
    ts.heads = np.load(st / "heads.npy", mmap_mode="r")
    ts.tok = tokens
    rng = np.random.default_rng(42)
    worst = 0.0
    for i in rng.choice(n, size=min(300, n), replace=False):
        s = ts[int(i)]
        for L, h in ((0, 0), (1, 2), (5, 7), (9, 6), (10, 6), (11, 2), (11, 11)):
            worst = max(worst, abs(TS.TokenSample.pooled_cosine(s, L, h) - df.loc[int(i), f"cos_L{L}H{h}"]))
    checks["stored_fp16_recompute_max_abs_diff"] = worst
    if worst > 5e-3:
        raise AssertionError(f"stored fp16 tokens do not reproduce the cosines: {worst}")
    checks["seconds"] = time.time() - t0
    checks["disk_gib"] = sum(f.stat().st_size for f in st.iterdir() if f.is_file()) / 2**30
    (st / "checks.json").write_text(json.dumps(checks, indent=2, default=str))
    write_status(st, {"status": "success", "finished": now(), "attacks": attacks,
                      "docs_per_group": int(tc["docs_per_group"]), "seed": int(tc["seed"])})
    print(json.dumps(checks, indent=1, default=str))
    print(f"[22] -> {st}")


if __name__ == "__main__":
    main()
