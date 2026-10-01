"""
scripts/01_run_repr_change.py
==============================
Exp 18 stage 01 — full-population encoder REPRESENTATION CHANGE, padded control -> attacked input
(exp18lib/repr_change.py documents states, regions and metrics).

Population: every aligned instance of the Exp 16 Exp 16 stage-17 paired manifest (judged DL19 pairs, qrel 2/3 or 0,
x 105 attacks; the 104 Exp 16 stage-17 alignment failures stay excluded and are re-reported here). Scores and
success labels are REUSED from Exp 16 stage 18 (18_paired_forward; successful <=> delta_score > 0); this stage
recomputes the score in the same forward only as a check (atol validation.score_atol).

One forward per batch of `run.batch_pairs` pairs: controls and their attacks in the SAME batch
(length-sorted within an attack). Hooks on 14 encoder states reduce every state immediately to
  * per-instance pooled metrics (4 regions) and token-wise summaries (3 aligned regions)
  * linear-CKA sufficient statistics (exp18lib.repr_change.CKAAccumulator, float64, never written)
No hidden state is stored (storage decision: per-instance scalars only, see README).

Checks (fail loudly): manifest re-encoding identical; per-pair region / token-id / mask integrity
(repr_change.check_pair, check_batch); shared regions at the embedding state have 1 - cos == 0 and
normalized L2 == 0 exactly; fresh scores == Exp 16 stage 18; no near-zero control norms.

Outputs: <outputs>/01_repr_change/
  per_attack/{attack}.parquet   long table: one row per (instance, state, region)
  cka.csv                       linear CKA raw + query-centred, per state x region x success x relevance group
  population.json               population actually used (counts, exclusions)
  checks.json, status.json
Not resumable per attack (CKA statistics span the whole population): rerun with --force.
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
import torch

EXP_DIR = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EXP_DIR))
import exp18lib  # noqa: E402,F401

from exp16lib import paired as P  # noqa: E402
from exp16lib.inputs import encode_attack_and_control  # noqa: E402
from exp16lib.run_utils import (file_sha256, is_already_successful, load_model, load_status, now,  # noqa: E402
                                read_jsonl, write_status)
from exp18lib import repr_change as R  # noqa: E402
from exp18lib.config import load_config, model_config, output_dir, resolve  # noqa: E402
from src.model_utils import build_monot5_input  # noqa: E402

STAGE_DIR = "01_repr_change"
META = ["attack_id", "attack_name", "attack_token", "attack_position", "repetitions", "pair_id", "qid", "docid",
        "qrel_grade", "relevance_group", "seq_len", "n_inserted", "alignment_boundary_shift"]
CATS = ["attack_name", "attack_token", "attack_position", "qid", "docid", "pair_id", "relevance_group", "state", "region"]


def long_table(meta: pd.DataFrame, M: dict) -> pd.DataFrame:
    """meta [N rows], M[k] arrays [N, S, 4] (pooled) / [N, S, 3] (token-wise) -> long [N * S * 4] rows."""
    N, S, G = len(meta), len(R.STATES), len(R.REGIONS)
    idx = np.repeat(np.arange(N), S * G)
    df = meta.iloc[idx].reset_index(drop=True)
    df["state"] = np.tile(np.repeat(R.STATES, G), N)
    df["layer"] = np.tile(np.repeat([R.STATE_LAYER[s] for s in R.STATES], G), N).astype(np.int8)
    df["region"] = np.tile(R.REGIONS, N * S)
    df["n_tokens_control"] = M["n_tok_ctl"].reshape(-1).astype(np.int16)
    df["n_tokens_attack"] = M["n_tok_atk"].reshape(-1).astype(np.int16)
    omc = M["one_minus_cosine"].reshape(-1)
    df["cosine"] = (1.0 - omc).astype(np.float32)
    df["one_minus_cosine"] = omc.astype(np.float32)
    df["normalized_l2"] = M["normalized_l2"].reshape(-1).astype(np.float32)

    def tw(k):                                           # token-wise: NaN for whole_full_visible
        a = np.full((N, S, G), np.nan)
        a[:, :, :R.N_TW] = M[k]
        return a.reshape(-1)
    df["tokenwise_mean_cosine"] = (1.0 - tw("tw_mean_omc")).astype(np.float32)
    df["tokenwise_median_cosine"] = (1.0 - tw("tw_q50_omc")).astype(np.float32)
    df["tokenwise_q10_cosine"] = (1.0 - tw("tw_q90_omc")).astype(np.float32)
    df["tokenwise_mean_one_minus_cosine"] = tw("tw_mean_omc").astype(np.float32)
    df["tokenwise_mean_normalized_l2"] = tw("tw_mean_nl2").astype(np.float32)
    df["tokenwise_median_normalized_l2"] = tw("tw_q50_nl2").astype(np.float32)
    df["tokenwise_q90_normalized_l2"] = tw("tw_q90_nl2").astype(np.float32)
    for c in CATS:
        df[c] = df[c].astype("category")
    return df


def main():
    p = argparse.ArgumentParser(description="Exp 18 stage 01: encoder representation change (control -> attack)")
    p.add_argument("--config", default=str(EXP_DIR / "configs" / "default.yaml"))
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    cfg = load_config(args.config)
    rc = cfg["run"]
    out = output_dir(cfg)
    stage = out / STAGE_DIR
    if not args.force and is_already_successful(stage, ["cka.csv", "population.json"]):
        print(f"[01] already complete in {stage} (use --force)")
        return
    t_start = time.time()
    write_status(stage, {"status": "running", "started": now()})
    e16 = resolve(cfg, cfg["exp16_outputs_dir"])
    mdir = e16 / P.MANIFEST_DIR
    mst = load_status(mdir)
    if file_sha256(mdir / "base_pairs.jsonl.gz") != mst["sha256"]["base_pairs"]:
        raise RuntimeError("stage-17 base manifest changed")
    base = {r["pair_id"]: r for r in read_jsonl(mdir / "base_pairs.jsonl.gz")}
    fwd = e16 / "18_paired_forward"                          # primary-scope forward (scores are scope-independent)
    attacks = list(mst["sha256"]["attacks"])
    if rc.get("max_attacks") is not None:                    # smoke only: evenly spaced subset
        k = int(rc["max_attacks"])
        attacks = [attacks[i] for i in np.linspace(0, len(attacks) - 1, k).round().astype(int)]
    max_inst = rc.get("max_instances_per_attack")
    qids = json.loads((mdir / "fold_map.json").read_text())["qids"]
    qindex = {q: i for i, q in enumerate(qids)}

    mcfg = model_config(cfg)
    model, tok, true_id, false_id, device = load_model(mcfg)
    max_len = int(mcfg["model"]["max_length"])
    bp = int(rc["batch_pairs"])
    acc = R.CKAAccumulator(len(R.STATES), len(qids), model.config.d_model, device)
    (stage / "per_attack").mkdir(parents=True, exist_ok=True)
    checks = Counter()
    max_emb_shared = 0.0
    max_score_diff = 0.0
    min_ctl_norm = np.inf
    pop = []

    for ai, a in enumerate(attacks):
        t0 = time.time()
        path = mdir / "attacks" / f"{a}.jsonl.gz"
        if file_sha256(path) != mst["sha256"]["attacks"][a]:
            raise RuntimeError(f"{path} changed after stage 17")
        recs = read_jsonl(path)
        if max_inst is not None:
            recs = [recs[i] for i in np.unique(np.linspace(0, len(recs) - 1, int(max_inst)).round().astype(int))]
        sc = pd.read_csv(fwd / "per_attack" / a / "rows.csv.gz", dtype=P.DT,
                         usecols=["pair_id", "score_control", "score_attack", "delta_score", "successful"])
        sc = sc.set_index("pair_id")
        pairs = []
        for r in recs:
            b = base[r["pair_id"]]
            atk, ctl, info = encode_attack_and_control(tok, b["query"], b["passage"], r["attacked_passage"], max_len)
            if (info["seq_len"], info["n_inserted"], atk.n_doc_tokens, ctl.n_doc_tokens) != \
                    (r["seq_len"], r["n_inserted"], r["n_document_tokens_attack"], r["n_document_tokens_control"]):
                raise RuntimeError(f"{a} {r['pair_id']}: re-encoding differs from the Exp 16 stage-17 manifest")
            reg = R.regions_from_info(info)
            R.check_pair(atk, ctl, reg, clean_ids=tok.encode(build_monot5_input(b["query"], b["passage"]))
                         if len(pairs) < 3 else None)          # full clean-id check on the first pairs of each attack
            if len(reg.orig_doc) != info["n_passage_tokens"]:
                raise AssertionError("orig_doc size != passage size")
            if not r["alignment_boundary_shift"] and reg.orig_doc != [i for i in range(reg.n) if ctl.doc_mask[i]]:
                raise AssertionError("orig_doc != control document mask on a non-boundary-shift pair")
            pairs.append((atk, ctl, reg))
        meta = pd.DataFrame([{m: r[m] for m in META} for r in recs])
        s = sc.loc[meta.pair_id]
        if s.index.duplicated().any() or len(s) != len(meta):
            raise RuntimeError(f"{a}: Exp 16 stage-18 score rows do not match the manifest")
        for c in ("score_control", "score_attack", "delta_score"):
            meta[c] = s[c].values.astype(np.float64)
        meta["successful"] = s["successful"].astype(bool).values
        if not (meta.successful.values == (meta.delta_score.values > 0)).all():
            raise RuntimeError("Exp 16 stage-18 successful != delta_score > 0")
        rel = (meta.relevance_group.values == R.RELEVANT).astype(np.int64)
        cell_all = torch.tensor(2 * meta.successful.values.astype(np.int64) + rel, device=device)
        q_all = torch.tensor([qindex[q] for q in meta.qid], device=device)

        N, S = len(pairs), len(R.STATES)
        M = {k: np.full((N, S, 4), np.nan) for k in ("one_minus_cosine", "normalized_l2", "n_tok_ctl", "n_tok_atk")}
        for k in ("tw_mean_omc", "tw_q50_omc", "tw_q90_omc", "tw_mean_nl2", "tw_q50_nl2", "tw_q90_nl2"):
            M[k] = np.full((N, S, R.N_TW), np.nan)
        order = sorted(range(N), key=lambda i: pairs[i][2].n)
        fresh = np.full((N, 2), np.nan)
        for start in range(0, N, bp):
            idx = order[start:start + bp]
            it = torch.tensor(idx, device=device)
            batch = R.collate_pairs([pairs[i] for i in idx], tok.pad_token_id, device)
            R.check_batch(batch)
            cell, qi = cell_all[it], q_all[it]
            res = {}

            def sink(si, m):
                acc.update(si, m["X"], m["Y"], cell, qi)
                res[si] = {k: v for k, v in m.items() if k not in ("X", "Y")}
            with torch.inference_mode(), R.StateCapture(model.encoder, sink) as cap:
                cap.set_batch(batch)
                B2 = batch["input_ids"].shape[0]
                dec = torch.full((B2, 1), model.config.decoder_start_token_id, dtype=torch.long, device=device)
                logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                               decoder_input_ids=dec).logits[:, 0, :]
                score = (logits[:, true_id] - logits[:, false_id]).double().cpu().numpy()
                if cap.seen != set(range(S)):
                    raise RuntimeError(f"states not captured: {set(range(S)) - cap.seen}")
            acc.add_counts(cell, qi)
            B = len(idx)
            fresh[idx, 0], fresh[idx, 1] = score[:B], score[B:]
            cnt = batch["region_mask"].sum(1).cpu().numpy()                     # [B, 3]
            n_full = batch["attack_full_mask"].sum(1).cpu().numpy()
            for si in range(S):
                m = res[si]
                M["one_minus_cosine"][idx, si] = m["one_minus_cosine"].cpu().numpy()
                M["normalized_l2"][idx, si] = m["normalized_l2"].cpu().numpy()
                for k in ("tw_mean_omc", "tw_q50_omc", "tw_q90_omc", "tw_mean_nl2", "tw_q50_nl2", "tw_q90_nl2"):
                    M[k][idx, si] = m[k].cpu().numpy()
                min_ctl_norm = min(min_ctl_norm, float(m["ctl_norm_min"]))
                checks["n_token_near_zero_control_norm"] += m["n_tok_small_norm"]
            M["n_tok_ctl"][idx] = np.concatenate([cnt, cnt[:, 2:3]], 1)[:, None, :]
            M["n_tok_atk"][idx] = np.concatenate([cnt, n_full[:, None]], 1)[:, None, :]
            checks["n_batches"] += 1
        # ---- per-attack checks --------------------------------------------------------------
        e = R.STATES.index("embedding")
        emb = max(np.abs(M["one_minus_cosine"][:, e, :3]).max(), np.abs(M["normalized_l2"][:, e, :3]).max(),
                  np.abs(M["tw_mean_omc"][:, e]).max())
        max_emb_shared = max(max_emb_shared, float(emb))
        if emb > float(cfg["validation"]["embedding_shared_atol"]):
            raise AssertionError(f"{a}: embedding-state shared regions differ ({emb}) -> misaligned pair")
        d = float(np.abs(fresh - meta[["score_control", "score_attack"]].values).max())
        max_score_diff = max(max_score_diff, d)
        if d > float(cfg["validation"]["score_atol"]):
            raise AssertionError(f"{a}: fresh score differs from stage 18 by {d}")
        for k in M:
            if k.startswith("tw") or k in ("one_minus_cosine", "normalized_l2"):
                if not np.isfinite(M[k]).all():
                    raise FloatingPointError(f"{a}: non-finite {k}")
        df = long_table(meta, M)
        df.to_parquet(stage / "per_attack" / f"{a}.parquet", index=False, compression="zstd")
        pop.append({"attack_name": a, "n_instances": N, "n_successful": int(meta.successful.sum()),
                    "n_relevant": int(rel.sum()), "n_boundary_shift": int(meta.alignment_boundary_shift.sum()),
                    "max_fresh_score_diff": d, "seconds": time.time() - t0})
        el = time.time() - t_start
        print(f"[01] {a:24s} {N} inst ({ai + 1}/{len(attacks)}) {time.time() - t0:.1f}s "
              f"ETA {el / (ai + 1) * (len(attacks) - ai - 1) / 60:.1f} min  score diff {d:.2e}", flush=True)

    # ---- CKA ------------------------------------------------------------------------------------
    rows = []
    for sg in R.SUCCESS_GROUPS:
        for rg in R.RELEVANCE_GROUPS:
            for r in acc.cka_table(R.cells_of(sg, rg)):
                rows.append({**r, "population": "control_vs_attack", "success_group": sg, "relevance_group": rg})
    cka = pd.DataFrame(rows)
    cka.to_csv(stage / "cka.csv", index=False)
    # the 104 Exp 16 stage-17 alignment failures (excluded) and the used population
    fails = pd.read_csv(mdir / "alignment_failures.csv", dtype=P.DT)
    popdf = pd.DataFrame(pop)
    popdf.to_csv(stage / "per_attack_population.csv", index=False)
    counts = acc.counts.cpu().numpy()
    popj = {"n_attacks": len(attacks), "n_instances": int(popdf.n_instances.sum()),
            "n_successful": int(popdf.n_successful.sum()), "n_relevant": int(popdf.n_relevant.sum()),
            "n_boundary_shift": int(popdf.n_boundary_shift.sum()),
            "n_queries": int((counts.sum(0) > 0).sum()),
            "instances_by_cell": {f"{'succ' if c // 2 else 'unsucc'}_{'rel' if c % 2 else 'nonrel'}": int(counts[c].sum())
                                  for c in range(4)},
            "exp16_stage17_alignment_failures_excluded": int(len(fails)) if max_inst is None and rc.get("max_attacks") is None else "n/a (smoke)",
            "states": R.STATES, "regions": R.REGIONS}
    (stage / "population.json").write_text(json.dumps(popj, indent=1))
    chk = {"max_embedding_shared_region_change": max_emb_shared, "max_fresh_vs_stage18_score_diff": max_score_diff,
           "min_pooled_control_norm": min_ctl_norm, **{k: int(v) for k, v in checks.items()},
           "cka_finite": bool(np.isfinite(cka[["cka_raw", "cka_query_centered"]].values).all())}
    (stage / "checks.json").write_text(json.dumps(chk, indent=1))
    write_status(stage, {"status": "success", "finished": now(), "seconds": time.time() - t_start,
                         "device": str(device), **{k: popj[k] for k in ("n_attacks", "n_instances")}})
    print(json.dumps(popj, indent=1))
    print(json.dumps(chk, indent=1))
    print(f"[01] done in {(time.time() - t_start) / 60:.1f} min -> {stage}")


if __name__ == "__main__":
    main()
