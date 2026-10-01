"""
scripts/01_run_head_change.py
==============================
Exp 19 stage 01 — per-head control -> attack change of the pre-o_proj head-output representation
(exp19lib/head_change.py), all 144 encoder heads, query + original-document regions.

Population: EXACTLY Exp 18's (outputs/01_repr_change, read-only): instance order, qid/docid/attack, qrel group,
scores and success labels are read from the Exp 18 per-attack Parquet; the attacked texts from the Exp 16
stage-17 manifest (the same records Exp 18 encoded). Counts are asserted against Exp 18 population.json.

One forward per batch of `run.batch_pairs` control/attack pairs (same batch, exp18lib.repr_change.collate_pairs);
pre-hooks on every SelfAttention.o reduce each layer immediately (no activation kept).

Validation (fail loudly, recorded in checks.json):
  * exp18lib.repr_change.check_pair / check_batch on every pair / batch (token ids, masks, inserted excluded)
  * fresh score == Exp 18 / Exp 16 stage-18 score (atol validation.score_atol)
  * first batch of every attack: exp16lib.heads.EncoderHeadCapture attached to the SAME forward with the Exp 16
    query/document masks -> per-head query/document cosines == Exp 16 stage 18 (all_layers) cached values
    (atol validation.head_crosscheck_atol), and its captured o_proj input == ours (bitwise)
  * first batch of the run: attack rows replaced by the control rows -> 1 - cos, L2 == 0 (atol 1e-6)

Outputs (01_head_change/):
  cka.csv                        linear CKA raw + query-centred per layer x head x region x success x relevance group
  cka_stats.npz                  the float64 sufficient statistics (any other grouping can be recomputed exactly)
  instances.parquet              one row per instance (Exp 18 metadata + scores), row i <-> metrics row i
  metrics/{attack}.npy           [n, 12 layers, 12 heads, 2 regions, 4 metrics] float32 (storage.per_instance_metrics)
  pooled/{attack}.npy            [n, 2 conditions, 12, 12, 2, 64] float16 (storage.pooled_vectors; Option 2 only)
  population.json, checks.json, status.json
Not resumable per attack (CKA statistics span the population): rerun with --force.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from collections import Counter

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

EXP_DIR = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EXP_DIR))
import exp19lib  # noqa: E402,F401

from exp16lib import paired as P  # noqa: E402
from exp16lib.heads import EncoderHeadCapture, all_encoder_heads  # noqa: E402
from exp16lib.inputs import collate, encode_attack_and_control  # noqa: E402
from exp16lib.run_utils import (file_sha256, is_already_successful, load_model, load_status, now,  # noqa: E402
                                read_jsonl, write_status)
from exp18lib import repr_change as R  # noqa: E402
from exp19lib import head_change as HC  # noqa: E402
from exp19lib.config import load_config, model_config, output_dir, resolve  # noqa: E402

STAGE_DIR = "01_head_change"
META = ["attack_id", "attack_name", "attack_token", "attack_position", "repetitions", "pair_id", "qid", "docid",
        "qrel_grade", "relevance_group", "seq_len", "n_inserted", "alignment_boundary_shift",
        "score_control", "score_attack", "delta_score", "successful"]


def exp18_meta(e18: pathlib.Path, attack: str) -> pd.DataFrame:
    t = pq.read_table(e18 / "01_repr_change" / "per_attack" / f"{attack}.parquet", columns=META + ["state", "region"])
    df = t.to_pandas()
    per = len(R.STATES) * len(R.REGIONS)
    m = df.iloc[::per].reset_index(drop=True)
    if len(df) != per * len(m) or not ((m.state.astype(str) == R.STATES[0]) & (m.region.astype(str) == R.REGIONS[0])).all():
        raise RuntimeError(f"{attack}: unexpected Exp 18 row layout")
    for c in m.columns:
        if isinstance(m[c].dtype, pd.CategoricalDtype):
            m[c] = m[c].astype(str)
    return m[META]


def main():
    p = argparse.ArgumentParser(description="Exp 19 stage 01: per-head control -> attack representation change")
    p.add_argument("--config", default=str(EXP_DIR / "configs" / "default.yaml"))
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    cfg = load_config(args.config)
    rc, sto, val = cfg["run"], cfg["storage"], cfg["validation"]
    out = output_dir(cfg)
    stage = out / STAGE_DIR
    if not args.force and is_already_successful(stage, ["cka.csv", "population.json"]):
        print(f"[01] already complete in {stage} (use --force)")
        return
    t_start = time.time()
    write_status(stage, {"status": "running", "started": now()})
    e16 = resolve(cfg, cfg["exp16_outputs_dir"])
    e18 = resolve(cfg, cfg["exp18_outputs_dir"])
    mdir = e16 / P.MANIFEST_DIR
    mst = load_status(mdir)
    if file_sha256(mdir / "base_pairs.jsonl.gz") != mst["sha256"]["base_pairs"]:
        raise RuntimeError("Exp 16 stage-17 base manifest changed")
    base = {r["pair_id"]: r for r in read_jsonl(mdir / "base_pairs.jsonl.gz")}
    pop18 = json.loads((e18 / "01_repr_change" / "population.json").read_text())
    attacks = list(mst["sha256"]["attacks"])
    if len(attacks) != pop18["n_attacks"]:
        raise RuntimeError("attack grid differs from Exp 18")
    if rc.get("max_attacks") is not None:
        attacks = [attacks[i] for i in np.linspace(0, len(attacks) - 1, int(rc["max_attacks"])).round().astype(int)]
    max_inst = rc.get("max_instances_per_attack")
    qids = json.loads((mdir / "fold_map.json").read_text())["qids"]
    qindex = {q: i for i, q in enumerate(qids)}

    mcfg = model_config(cfg)
    model, tok, true_id, false_id, device = load_model(mcfg)
    max_len = int(mcfg["model"]["max_length"])
    enc = model.encoder
    nL, nH, dk = len(enc.block), enc.config.num_heads, enc.config.d_kv
    assert (nL, nH, dk) == (HC.N_LAYERS, HC.N_HEADS, HC.D_KV), (nL, nH, dk)
    all_heads = all_encoder_heads(range(nL))
    fwd_all = e16 / "18_paired_forward_all_layers" / "per_attack"
    acc = HC.HeadCKAAccumulator(nL, nH, dk, len(qids), device)
    for d in ("metrics",) + (("pooled",) if sto.get("pooled_vectors") else ()):
        (stage / d).mkdir(parents=True, exist_ok=True)
    checks = Counter()
    worst = {"score": 0.0, "exp16_head_cos": 0.0, "identical_input": 0.0}
    min_pool_norm = np.inf
    metas, pop = [], []

    for ai, a in enumerate(attacks):
        t0 = time.time()
        path = mdir / "attacks" / f"{a}.jsonl.gz"
        if file_sha256(path) != mst["sha256"]["attacks"][a]:
            raise RuntimeError(f"{path} changed after Exp 16 stage 17")
        recs = read_jsonl(path)
        meta = exp18_meta(e18, a)
        if [r["pair_id"] for r in recs] != meta.pair_id.tolist():
            raise RuntimeError(f"{a}: instance set/order differs from Exp 18")
        if max_inst is not None:
            keep = np.unique(np.linspace(0, len(recs) - 1, int(max_inst)).round().astype(int))
            recs = [recs[i] for i in keep]
            meta = meta.iloc[keep].reset_index(drop=True)
        pairs = []
        for r in recs:
            b = base[r["pair_id"]]
            atk, ctl, info = encode_attack_and_control(tok, b["query"], b["passage"], r["attacked_passage"], max_len)
            if (info["seq_len"], info["n_inserted"]) != (r["seq_len"], r["n_inserted"]):
                raise RuntimeError(f"{a} {r['pair_id']}: re-encoding differs from the manifest")
            reg = R.regions_from_info(info)
            R.check_pair(atk, ctl, reg)
            if len(reg.orig_doc) != info["n_passage_tokens"]:
                raise AssertionError("orig_doc size != passage size")
            pairs.append((atk, ctl, reg))
        rel = (meta.relevance_group.values == R.RELEVANT).astype(np.int64)
        succ = meta.successful.values.astype(bool)
        if not (succ == (meta.delta_score.values > 0)).all():
            raise RuntimeError("successful != delta_score > 0")
        cell_all = torch.tensor(2 * succ.astype(np.int64) + rel, device=device)
        q_all = torch.tensor([qindex[q] for q in meta.qid], device=device)
        N = len(pairs)
        M = np.full((N, nL, nH, HC.N_REG, len(HC.METRICS)), np.nan, np.float32)
        PV = np.zeros((N, 2, nL, nH, HC.N_REG, dk), np.float16) if sto.get("pooled_vectors") else None
        fresh = np.full((N, 2), np.nan)
        order = sorted(range(N), key=lambda i: pairs[i][2].n)
        bp = int(rc["batch_pairs"])
        for bi, start in enumerate(range(0, N, bp)):
            idx = order[start:start + bp]
            it = torch.tensor(idx, device=device)
            batch = R.collate_pairs([pairs[i] for i in idx], tok.pad_token_id, device)
            R.check_batch(batch)
            if ai == 0 and bi == 0:                                   # identical-input sanity check
                same = {k: v.clone() for k, v in batch.items()}
                B = len(idx)
                same["input_ids"][B:], same["attention_mask"][B:] = same["input_ids"][:B], same["attention_mask"][:B]
                res0 = {}
                with torch.inference_mode(), HC.HeadCapture(enc, lambda L, m: res0.__setitem__(L, m)) as cap0:
                    cap0.set_batch(same["region_mask"])
                    enc(input_ids=same["input_ids"], attention_mask=same["attention_mask"])
                d0 = max(max(float(res0[L][k].abs().max()) for k in HC.METRICS) for L in range(nL))
                ck = HC.cka_table_from_state({"Gxx": np.stack([torch.einsum("brhd,brhe->rhde", res0[L]["X"], res0[L]["X"]).cpu().numpy()[:, None] for L in range(nL)]),
                                              "Gyy": np.stack([torch.einsum("brhd,brhe->rhde", res0[L]["Y"], res0[L]["Y"]).cpu().numpy()[:, None] for L in range(nL)]),
                                              "Gxy": np.stack([torch.einsum("brhd,brhe->rhde", res0[L]["X"], res0[L]["Y"]).cpu().numpy()[:, None] for L in range(nL)]),
                                              "sx": np.stack([res0[L]["X"].sum(0).cpu().numpy()[:, None, None] for L in range(nL)]),
                                              "sy": np.stack([res0[L]["Y"].sum(0).cpu().numpy()[:, None, None] for L in range(nL)]),
                                              "counts": np.array([[float(B)]])}, [0])
                dcka = max(abs(1 - r["cka_raw"]) for r in ck)
                worst["identical_input"] = max(d0, dcka)
                if d0 > 1e-6 or dcka > 1e-6:
                    raise AssertionError(f"identical control/attack inputs give change {d0} / 1-CKA {dcka}")
            cell, qi = cell_all[it], q_all[it]
            res = {}

            def sink(L, m):
                acc.update(L, m["X"], m["Y"], cell, qi)
                res[L] = m
            xcheck = bi == 0
            with torch.inference_mode(), HC.HeadCapture(enc, sink, keep_inputs=xcheck) as cap:
                cap.set_batch(batch["region_mask"])
                ec = None
                if xcheck:                                             # Exp 16 extraction on the same forward
                    seqs = [pairs[i][1] for i in idx] + [pairs[i][0] for i in idx]
                    cb = collate(seqs, tok.pad_token_id, device)
                    if not (torch.equal(cb["input_ids"], batch["input_ids"])
                            and torch.equal(cb["attention_mask"], batch["attention_mask"])):
                        raise AssertionError("exp16 collate != exp18 collate_pairs")
                    ec = EncoderHeadCapture(enc, all_heads, keep_inputs=True).__enter__()
                    ec.set_masks(cb["query_mask"], cb["doc_mask"])
                try:
                    B2 = batch["input_ids"].shape[0]
                    dec = torch.full((B2, 1), model.config.decoder_start_token_id, dtype=torch.long, device=device)
                    logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                                   decoder_input_ids=dec).logits[:, 0, :]
                finally:
                    if ec is not None:
                        ec.__exit__()
                if cap.seen != set(range(nL)):
                    raise RuntimeError("not every layer captured")
                score = (logits[:, true_id] - logits[:, false_id]).double().cpu().numpy()
            acc.add_counts(cell, qi)
            B = len(idx)
            fresh[idx, 0], fresh[idx, 1] = score[:B], score[B:]
            for L in range(nL):
                m = res[L]
                M[idx, L] = torch.stack([m[k] for k in HC.METRICS], -1).permute(0, 2, 1, 3).float().cpu().numpy()
                min_pool_norm = min(min_pool_norm, m["pooled_ctl_norm_min"])
                checks["n_token_near_zero_control_norm"] += m["n_tok_small_norm"]
                if PV is not None:
                    PV[idx, 0, L] = m["X"].permute(0, 2, 1, 3).cpu().numpy().astype(np.float16)
                    PV[idx, 1, L] = m["Y"].permute(0, 2, 1, 3).cpu().numpy().astype(np.float16)
            if xcheck:
                for L in range(nL):
                    if not torch.equal(ec.inputs[L], cap.inputs[L]):
                        raise AssertionError(f"layer {L}: captured o_proj input differs from EncoderHeadCapture")
                hcols = [f"{sd}_head_{h.label}" for sd in ("control", "attack") for h in all_heads]   # 144 x 2
                if not (fwd_all / a / "rows.csv.gz").exists():
                    raise FileNotFoundError(f"Exp 16 stage 18 all_layers rows missing for {a}")
                old = pd.read_csv(fwd_all / a / "rows.csv.gz", dtype=P.DT, usecols=["pair_id"] + hcols)
                if True:
                    o = old.set_index("pair_id").loc[meta.pair_id.values[idx]]
                    mine = np.stack([ec.cos[h.label].cpu().numpy() for h in all_heads], 1)      # [2B, 144]
                    d = float(max(np.abs(mine[:B] - o[[f"control_head_{h.label}" for h in all_heads]].values).max(),
                                  np.abs(mine[B:] - o[[f"attack_head_{h.label}" for h in all_heads]].values).max()))
                    worst["exp16_head_cos"] = max(worst["exp16_head_cos"], d)
                    checks["n_exp16_head_crosschecks"] += 1
                    if d > float(val["head_crosscheck_atol"]):
                        raise AssertionError(f"{a}: head cosines differ from Exp 16 stage 18 all_layers by {d}")
            checks["n_batches"] += 1
        d = float(np.abs(fresh - meta[["score_control", "score_attack"]].values).max())
        worst["score"] = max(worst["score"], d)
        if d > float(val["score_atol"]):
            raise AssertionError(f"{a}: fresh score differs from Exp 18 by {d}")
        if not np.isfinite(M).all():
            raise FloatingPointError(f"{a}: non-finite head metric")
        if sto.get("per_instance_metrics"):
            np.save(stage / "metrics" / f"{a}.npy", M)
        if PV is not None:
            np.save(stage / "pooled" / f"{a}.npy", PV)
        meta["row_in_attack"] = np.arange(N)
        metas.append(meta)
        pop.append({"attack_name": a, "n_instances": N, "n_successful": int(succ.sum()), "n_relevant": int(rel.sum()),
                    "max_score_diff": d, "seconds": time.time() - t0})
        el = time.time() - t_start
        print(f"[01] {a:24s} {N} inst ({ai + 1}/{len(attacks)}) {time.time() - t0:.1f}s "
              f"ETA {el / (ai + 1) * (len(attacks) - ai - 1) / 60:.1f} min  score diff {d:.1e}  "
              f"head-cos diff {worst['exp16_head_cos']:.1e}", flush=True)

    st = acc.state()
    np.savez(stage / "cka_stats.npz", **st, qids=np.array(qids), regions=np.array(HC.REGIONS))
    rows = []
    for sg in R.SUCCESS_GROUPS:
        for rg in R.RELEVANCE_GROUPS:
            for r in HC.cka_table_from_state(st, R.cells_of(sg, rg)):
                rows.append({**r, "success_group": sg, "relevance_group": rg})
    cka = pd.DataFrame(rows)
    cka.to_csv(stage / "cka.csv", index=False, float_format="%.9g")
    inst = pd.concat(metas, ignore_index=True)
    inst.to_parquet(stage / "instances.parquet", index=False)
    popdf = pd.DataFrame(pop)
    popdf.to_csv(stage / "per_attack_population.csv", index=False)
    full = rc.get("max_attacks") is None and max_inst is None
    popj = {"n_attacks": len(attacks), "n_instances": len(inst), "n_queries": int(inst.qid.nunique()),
            "n_pairs": int(inst.pair_id.nunique()), "n_successful": int(inst.successful.sum()),
            "n_unsuccessful": int((~inst.successful.astype(bool)).sum()),
            "n_relevant": int((inst.relevance_group == R.RELEVANT).sum()),
            "n_boundary_shift": int(inst.alignment_boundary_shift.sum()), "smoke": not full}
    if full:
        for k in ("n_attacks", "n_instances", "n_queries", "n_successful", "n_relevant", "n_boundary_shift"):
            if popj[k] != pop18[k]:
                raise AssertionError(f"population {k} = {popj[k]} != Exp 18 {pop18[k]}")
        popj["matches_exp18"] = True
    (stage / "population.json").write_text(json.dumps(popj, indent=1))
    chk = {**{f"max_{k}_diff": v for k, v in worst.items()}, "min_pooled_control_norm": min_pool_norm,
           **{k: int(v) for k, v in checks.items()},
           "cka_finite": bool(np.isfinite(cka[["cka_raw", "cka_query_centered"]].values).all())}
    (stage / "checks.json").write_text(json.dumps(chk, indent=1))
    write_status(stage, {"status": "success", "finished": now(), "seconds": time.time() - t_start,
                         "device": str(device), "n_instances": len(inst)})
    print(json.dumps(popj, indent=1))
    print(json.dumps(chk, indent=1))
    print(f"[01] done in {(time.time() - t_start) / 60:.1f} min -> {stage}")


if __name__ == "__main__":
    main()
