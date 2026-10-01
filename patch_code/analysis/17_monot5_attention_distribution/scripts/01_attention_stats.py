"""
scripts/01_attention_stats.py
==============================
Stage 01 — encoder attention-distribution statistics on EXACTLY the Exp 16 stage-22 sample.

The cached stage-22 token arrays (input_ids, attention_mask, query/doc/inserted masks) are
re-batched (length-sorted, right-padded with attention 0) and run through monoT5 (fp32, eager
attention) with output_attentions=True: 12 encoder layers x [B, 12, S, S]. Per batch, every
statistic of exp17lib.attention_stats is reduced to one scalar per (sequence, source region,
key scope, metric, layer, head) and the attention matrices are discarded.

Validation (fails loudly, before anything is saved as success):
  counts / pairing identical to stage 22; mask integrity (query/doc disjoint, regions visible,
  control insertion slots = masked pads outside every region, inserted region = exactly the
  injected string); attention shape; masked keys (control slots, batch padding) receive exactly
  0 attention; visible rows sum to 1; recomputed monoT5 score == stage-22 score for EVERY
  sequence and pre-o_proj head outputs == stage-22 heads.npy for the first batches (same inputs);
  direct NumPy recomputation of random cells; value ranges; synthetic uniform / one-hot /
  uniform-region-mass checks; shared_key == full_visible for clean and control sequences.

Outputs (outputs/01_attention_stats/):
  attention_statistics.npz     R [n, region, scope, metric, 144], M [n, direction, (raw, norm), 144]
  attention_statistics.csv.gz  long: one row per (sequence, head, source_region, key_scope) with metric columns
                               (clean/control: regions query, doc; scope full_visible — shared_key is
                               identical by construction and validated; attack: 4 regions x 2 scopes)
  injected_mass.csv.gz         attacked sequences: one row per (sequence, head), raw + normalised mass per direction
  sequences_meta.csv           identifiers, scores, key / region token counts
  validation.json, mask_inspection.txt, status.json
"""

from __future__ import annotations

import json
import sys
import time
import pathlib

import numpy as np
import pandas as pd
import torch

EXP_DIR = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EXP_DIR))
import exp17lib  # noqa: E402,F401

from exp16lib.metrics_screen import region_masks  # noqa: E402
from exp16lib.run_utils import is_already_successful, load_model, now, write_status  # noqa: E402
from exp16lib.token_sample import TokenSample  # noqa: E402
from exp17lib import attention_stats as AT  # noqa: E402
from exp17lib.config import load_config, model_config, output_dir, resolve  # noqa: E402

STAGE = "01_attention_stats"
LABELS = [f"L{L}H{h}" for L in range(12) for h in range(12)]
TEMPLATE_PIECES = ["▁", "Query", ":", "▁Document", "▁Relevan", "t", "</s>"]
KEEP_META = ["seq_id", "kind", "attack_name", "pair_id", "qid", "docid", "qrel_grade", "relevance_group",
             "attack_token", "attack_position", "repetitions", "score", "delta_score", "successful", "n_tokens"]


def _argparser():
    import argparse
    p = argparse.ArgumentParser(description="Exp 17 stage 01: encoder attention-distribution statistics")
    p.add_argument("--config", default=str(EXP_DIR / "configs" / "default.yaml"))
    p.add_argument("--force", action="store_true")
    return p


def _synthetic_checks() -> dict:
    """Checks 12-14 on synthetic matrices through the production code path."""
    S = 20
    vis = torch.ones(1, S, dtype=torch.bool)
    vis[0, [4, 11]] = False
    U = torch.where(vis[:, None, None, :], 1.0 / vis.sum(), 0.0).expand(1, 1, S, S).float()
    su = AT.row_stats(U, vis, torch.tensor([S]))
    O = torch.zeros(1, 1, S, S)
    O[0, 0, torch.arange(S), (torch.arange(S) * 7) % S] = 1.0
    so = AT.row_stats(O, torch.ones(1, S, dtype=torch.bool), torch.tensor([S]))
    src = torch.zeros(1, S, dtype=torch.bool); src[0, :5] = True
    tgt = torch.zeros(1, S, dtype=torch.bool); tgt[0, 12:17] = True
    _, norm = AT.region_mass(U, vis, src, tgt)
    return {"uniform_entropy_norm_max_abs_dev_from_1": float((su["entropy_norm"] - 1).abs().max()),
            "one_hot_entropy_norm_max_abs": float(so["entropy_norm"].abs().max()),
            "uniform_region_mass_norm": float(norm.item())}


def main():
    args = _argparser().parse_args()
    cfg = load_config(args.config)
    out = output_dir(cfg)
    st = out / STAGE
    if is_already_successful(st, ["attention_statistics.npz"]) and not args.force:
        print(f"[01] {st} already complete (use --force)")
        return
    st.mkdir(parents=True, exist_ok=True)
    write_status(st, {"status": "running", "started": now()})
    t0 = time.time()
    vc = cfg["validation"]
    ts = TokenSample(resolve(cfg, cfg["token_sample_dir"]))
    meta_all = ts.meta
    val: dict = {"token_sample_dir": str(ts.dir)}

    # ---- selection (smoke: first N base pairs) ----------------------------------------------------------
    if cfg["sample"]["max_pairs"]:
        keep = list(dict.fromkeys(meta_all.pair_id))[: int(cfg["sample"]["max_pairs"])]
        sel = meta_all.index[meta_all.pair_id.isin(keep)].values
    else:
        sel = meta_all.index.values
    meta = meta_all.loc[sel].reset_index(drop=True)
    n = len(meta)
    kinds = meta.kind.values

    # ---- validation 1: counts + pairing (15) ----------------------------------------------------------------
    atk = meta[meta.kind == "attack"]
    c = {"n_sequences": n, "by_kind": meta.kind.value_counts().to_dict(), "n_pairs": int(meta.pair_id.nunique()),
         "n_queries": int(meta.qid.nunique()),
         "clean_by_group": meta[meta.kind == "clean"].relevance_group.value_counts().to_dict(),
         "qrel_grades": sorted(int(g) for g in meta.qrel_grade.unique()),
         "n_successful_attacks": int((atk.successful == True).sum()),  # noqa: E712
         "n_successful_by_group": atk[atk.successful == True].relevance_group.value_counts().to_dict()}  # noqa: E712
    val["counts"] = c
    ok = 1 not in c["qrel_grades"] and (meta_all.seq_id.values == np.arange(len(meta_all))).all()
    if cfg["sample"]["expect_full_sample"]:
        ok &= (n == 6325 and c["n_pairs"] == 253 and c["n_queries"] == 43
               and c["by_kind"] == {"control": 3036, "attack": 3036, "clean": 253}
               and c["clean_by_group"] == {"relevant": 126, "nonrelevant": 127} and c["n_successful_attacks"] == 1026)
    ctl = meta[meta.kind == "control"]
    key_a = list(zip(atk.attack_name, atk.pair_id))
    key_c = list(zip(ctl.attack_name, ctl.pair_id))
    pair_ok = len(set(key_a)) == len(key_a) == len(key_c) == len(set(key_c)) and set(key_a) == set(key_c)
    cmap = dict(zip(key_c, ctl.index))
    ic_of_a = np.array([cmap[k] for k in key_a]) if pair_ok else np.array([], int)
    bad_pair = 0
    for ia, ic in zip(atk.index, ic_of_a):
        sa, sc = ts[int(meta.seq_id[ia])], ts[int(meta.seq_id[ic])]
        ins = sa["inserted_mask"]
        bad_pair += int(len(ins) != len(sc["inserted_mask"]) or not np.array_equal(ins, sc["inserted_mask"])
                        or not np.array_equal(sa["input_ids"][~ins], sc["input_ids"][~ins])
                        or sc["attention_mask"][ins].any() or not sa["attention_mask"].all()
                        or abs((meta.score[ia] - meta.score[ic]) - meta.delta_score[ia]) > 1e-6
                        or meta.delta_score[ia] != meta.delta_score[ic]
                        or bool(meta.successful[ia]) != bool(meta.delta_score[ia] > 0)
                        or any(meta.loc[ia, k] != meta.loc[ic, k] for k in ("qid", "docid", "relevance_group")))
    val["pairing"] = {"one_control_per_attack": bool(pair_ok), "n_bad_pairs": bad_pair}
    ok &= pair_ok and bad_pair == 0
    val["counts_ok"] = bool(ok)
    if not ok:
        (st / "validation.json").write_text(json.dumps(val, indent=2, default=str))
        raise SystemExit(f"STOP: stage-22 counts / pairing check failed: {val}")

    # ---- validation 5/6: masks ---------------------------------------------------------------------------------
    c16 = model_config(cfg)
    model, tok, tid, fid, dev = load_model(c16)
    if getattr(model.config, "_attn_implementation", "eager") != "eager":
        model.set_attn_implementation("eager")
    val["attn_implementation"] = model.config._attn_implementation
    pad, colon = tok.pad_token_id, tok.convert_tokens_to_ids(":")
    template = set(tok.convert_tokens_to_ids(TEMPLATE_PIECES)) | {pad}
    mk = {"query_doc_overlap": 0, "region_not_visible": 0, "control_slot_in_region_or_visible": 0,
          "control_slot_not_pad": 0, "clean_has_inserted": 0, "attack_inserted_outside_doc_or_invisible": 0,
          "non_template_outside_regions": 0, "attack_word_count_ne_repetitions": 0,
          "attack_inserted_other_tokens": 0, "attack_orig_partition_broken": 0}
    strings = {}
    for r in range(n):
        s = ts[int(meta.seq_id[r])]
        q, d, ins, am, ids = s["query_mask"], s["doc_mask"], s["inserted_mask"], s["attention_mask"], s["input_ids"]
        mk["query_doc_overlap"] += int((q & d).sum())
        mk["region_not_visible"] += int(((q | d) & ~am).sum())
        mk["non_template_outside_regions"] += int((~np.isin(ids[~(q | d)], list(template))).sum())
        if kinds[r] == "clean":
            mk["clean_has_inserted"] += int(ins.sum())
        elif kinds[r] == "control":
            mk["control_slot_in_region_or_visible"] += int((ins & (q | d | am)).sum())
            mk["control_slot_not_pad"] += int((ids[ins] != pad).sum())
        else:
            reg = region_masks(q, d, ins, "attack")
            mk["attack_inserted_outside_doc_or_invisible"] += int((ins & (~d | ~am)).sum())
            mk["attack_orig_partition_broken"] += int((reg["orig"] & ins).sum()) + int((reg["orig"] | reg["ins"]).sum() != d.sum())
            pieces = tok.convert_ids_to_tokens(ids[ins].tolist())
            word = "▁" + str(meta.attack_token[r])
            mk["attack_word_count_ne_repetitions"] += int(sum(p == word for p in pieces) != int(meta.repetitions[r]))
            mk["attack_inserted_other_tokens"] += int(any(p != word and t_ != colon for p, t_ in zip(pieces, ids[ins])))
            strings.setdefault(meta.attack_name[r], set()).add(tuple(ids[ins].tolist()))
    mk["inserted_string_identical_within_attack"] = {k: len(v) == 1 for k, v in strings.items()}
    mk["inserted_strings"] = {k: tok.decode(list(next(iter(v)))) for k, v in strings.items()}
    val["masks"] = mk
    ok_mask = all(v == 0 for k, v in mk.items() if isinstance(v, int)) and all(mk["inserted_string_identical_within_attack"].values())
    val["masks_ok"] = bool(ok_mask)
    if not ok_mask:
        (st / "validation.json").write_text(json.dumps(val, indent=2, default=str))
        raise SystemExit(f"STOP: mask validation failed: {mk}")
    print(f"[01] counts / pairing / masks OK ({time.time() - t0:.0f}s)", flush=True)

    # ---- forward + statistics ----------------------------------------------------------------------------------
    R = np.full((n, len(AT.REGIONS), len(AT.SCOPES), len(AT.METRICS), 144), np.nan, np.float32)
    M = np.full((n, len(AT.MASS_DIRS), 2, 144), np.nan, np.float32)
    score = np.full(n, np.nan)
    masked_key_max = np.full(n, np.nan)          # max attention any row gives to an invisible key (control slots / padding)
    row_sum_dev = 0.0
    shape_ok = True
    pre_o_worst, pre_o_n = 0.0, 0
    direct_worst, direct_n = 0.0, 0
    rng = np.random.default_rng(7)
    bs = int(cfg["runtime"]["batch_size"])
    order = np.argsort(meta.n_tokens.values, kind="stable")
    start_id = model.config.decoder_start_token_id
    for bi, start in enumerate(range(0, n, bs)):
        idx = order[start:start + bs]
        seqs = [ts[int(meta.seq_id[r])] for r in idx]
        S = max(len(s["input_ids"]) for s in seqs)

        def padded(key, v=0):
            return torch.tensor(np.stack([np.pad(s[key].astype(np.int64), (0, S - len(s[key])), constant_values=v)
                                          for s in seqs]), device=dev)
        ids, am = padded("input_ids", pad), padded("attention_mask")
        qm, dm, im = padded("query_mask"), padded("doc_mask"), padded("inserted_mask")
        ntok = torch.tensor([len(s["input_ids"]) for s in seqs], device=dev)
        cap, handles = {}, []
        if bi < int(vc["pre_o_n_batches"]):
            for L in range(12):
                handles.append(model.encoder.block[L].layer[0].SelfAttention.o.register_forward_pre_hook(
                    lambda mod, a, L=L: cap.__setitem__(L, a[0].detach().float().cpu())))
        with torch.inference_mode():
            o = model(input_ids=ids, attention_mask=am,
                      decoder_input_ids=torch.full((len(idx), 1), start_id, dtype=torch.long, device=dev),
                      output_attentions=True, use_cache=False)
        for h in handles:
            h.remove()
        atts = o.encoder_attentions
        lg = o.logits[:, 0, :]
        score[idx] = (lg[:, tid] - lg[:, fid]).double().cpu().numpy()
        shape_ok &= len(atts) == 12 and all(tuple(a.shape) == (len(idx), 12, S, S) for a in atts)
        vis = am.bool()
        for a in atts:
            row_sum_dev = max(row_sum_dev, float(((a.sum(-1) - 1).abs() * vis[:, None, :]).max()))
        inv = (~vis)[:, None, None, :]
        mx = torch.stack([(a * inv).amax(dim=(1, 2, 3)) for a in atts]).amax(0)
        masked_key_max[idx] = torch.where((~vis).any(-1), mx, torch.zeros_like(mx)).cpu().numpy()
        masks = AT.batch_masks(am, qm, dm, im)
        Rb, Mb = AT.batch_statistics(atts, masks, ntok, int(cfg["runtime"]["local_window"]))
        R[idx] = Rb.reshape(len(idx), *Rb.shape[1:4], 144)
        M[idx] = Mb.reshape(len(idx), *Mb.shape[1:3], 144)

        if cap:                                           # same inputs as stage 22: pre-o_proj head outputs
            for j, s in enumerate(seqs):
                T_ = len(s["input_ids"])
                stored = np.asarray(s["heads"], np.float32).reshape(T_, 12, 768)
                for L in range(12):
                    x = cap[L][j, :T_].numpy()
                    pre_o_worst = max(pre_o_worst, float((np.abs(x - stored[:, L]) / np.maximum(np.abs(x), 1)).max()))
                    pre_o_n += 1
        if bi < int(vc["direct_check_batches"]):          # direct NumPy recomputation of random cells
            for _ in range(int(vc["direct_check_cells"])):
                j, L, h = int(rng.integers(len(idx))), int(rng.integers(12)), int(rng.integers(12))
                r = idx[j]
                s = seqs[j]
                T_ = len(s["input_ids"])
                Aj = atts[L][j, h, :T_, :T_].double().cpu().numpy()
                regs = {"query": s["query_mask"], "doc": s["doc_mask"],
                        "orig": s["doc_mask"] & ~s["inserted_mask"], "ins": s["doc_mask"] & s["inserted_mask"]}
                keys = {"full_visible": s["attention_mask"], "shared_key": s["attention_mask"] & ~s["inserted_mask"]}
                reg = list(regs)[int(rng.integers(4))]
                sc = list(keys)[int(rng.integers(2))]
                rows = np.where(regs[reg])[0]
                if len(rows) == 0:
                    continue
                refs = [AT.row_stats_np(Aj[i], keys[sc], i, T_, int(cfg["runtime"]["local_window"])) for i in rows]
                for mi, m in enumerate(AT.METRICS):
                    v = np.nanmean([x[m] for x in refs])
                    got = float(R[r, AT.REGIONS.index(reg), AT.SCOPES.index(sc), mi, L * 12 + h])
                    direct_worst = max(direct_worst, abs(got - v) / max(abs(v), 1e-3))
                    direct_n += 1
                if kinds[r] == "attack":
                    for di, (a_, b_) in enumerate(AT.MASS_DIRS):
                        rw, nm = AT.region_mass_np(Aj, regs[a_], regs[b_], s["attention_mask"])
                        for k_, v in ((0, rw), (1, nm)):
                            got = float(M[r, di, k_, L * 12 + h])
                            direct_worst = max(direct_worst, abs(got - v) / max(abs(v), 1e-3))
                            direct_n += 1
        del atts, o
        if bi % 25 == 0:
            print(f"[01] batch {bi} ({min(start + bs, n)}/{n} sequences, S={S}) {time.time() - t0:.0f}s", flush=True)

    # ---- post-hoc validation ------------------------------------------------------------------------------------
    MI = {m: i for i, m in enumerate(AT.METRICS)}
    na = kinds != "attack"
    en = R[:, :, :, MI["entropy_norm"]]
    tol = 1e-4
    val["forward"] = {
        "attention_shape_ok": bool(shape_ok), "n_layers": 12, "heads_per_layer": 12,
        "score_max_abs_diff_vs_stage22": float(np.abs(score - meta.score.values).max()),
        "masked_key_max_attention": float(np.nanmax(masked_key_max)),
        "n_sequences_with_masked_keys_in_own_length": int((meta.kind == "control").sum()),
        "visible_row_sum_max_abs_dev": row_sum_dev,
        "pre_o_vs_stage22_max_rel_diff": pre_o_worst, "pre_o_n_layer_sequence_checks": pre_o_n,
        "direct_numpy_max_rel_diff": direct_worst, "direct_numpy_n_values": direct_n}
    val["ranges"] = {
        "entropy_norm_min": float(np.nanmin(en)), "entropy_norm_max": float(np.nanmax(en)),
        "top_order_violations": int((R[..., MI["max_attn"], :] > R[..., MI["top3"], :] + tol).sum()
                                    + (R[..., MI["top3"], :] > R[..., MI["top5"], :] + tol).sum()
                                    + (R[..., MI["top5"], :] > 1 + tol).sum()),
        "neff_below_1": int((R[..., MI["neff"], :] < 1 - tol).sum()),
        "neff_norm_out_of_(0,1]": int(((R[..., MI["neff_norm"], :] <= 0) | (R[..., MI["neff_norm"], :] > 1 + tol)).sum()),
        "dist_norm_out_of_[0,1]": int(((R[..., MI["dist_norm"], :] < -tol) | (R[..., MI["dist_norm"], :] > 1 + tol)).sum()),
        "nan_in_defined_cells": int(np.isnan(R[:, :3]).sum() + np.isnan(R[~na, 3]).sum()),
        "shared_equals_full_for_clean_control": bool(np.array_equal(R[na, :, 0], R[na, :, 1], equal_nan=True)),
        "orig_equals_doc_for_clean_control": bool(np.array_equal(R[na, 2], R[na, 1], equal_nan=True)),
        "ins_nan_for_clean_control": bool(np.isnan(R[na, 3]).all() and np.isnan(M[na][:, :4]).all()),
        "mass_nan_in_attack": int(np.isnan(M[~na]).sum())}
    val["synthetic"] = _synthetic_checks()
    fv, rg, sy = val["forward"], val["ranges"], val["synthetic"]
    ok_all = (fv["attention_shape_ok"] and fv["score_max_abs_diff_vs_stage22"] <= float(vc["score_atol"])
              and fv["masked_key_max_attention"] == 0.0 and fv["visible_row_sum_max_abs_dev"] <= float(vc["row_sum_atol"])
              and fv["pre_o_vs_stage22_max_rel_diff"] <= float(vc["pre_o_rel_tol"]) and fv["pre_o_n_layer_sequence_checks"] > 0
              and fv["direct_numpy_max_rel_diff"] <= float(vc["direct_rtol"]) and fv["direct_numpy_n_values"] > 0
              and rg["entropy_norm_min"] >= -tol and rg["entropy_norm_max"] <= 1 + tol and rg["top_order_violations"] == 0
              and rg["neff_below_1"] == 0 and rg["neff_norm_out_of_(0,1]"] == 0 and rg["dist_norm_out_of_[0,1]"] == 0
              and rg["nan_in_defined_cells"] == 0 and rg["shared_equals_full_for_clean_control"]
              and rg["orig_equals_doc_for_clean_control"] and rg["ins_nan_for_clean_control"] and rg["mass_nan_in_attack"] == 0
              and sy["uniform_entropy_norm_max_abs_dev_from_1"] < 1e-5 and sy["one_hot_entropy_norm_max_abs"] < 1e-6
              and abs(sy["uniform_region_mass_norm"] - 1) < 1e-5)
    val["all_ok"] = bool(ok_all)
    (st / "validation.json").write_text(json.dumps(val, indent=2, default=str))
    if not ok_all:
        raise SystemExit(f"STOP: attention validation failed, see {st / 'validation.json'}")
    print(f"[01] forward + validation OK ({time.time() - t0:.0f}s)", flush=True)

    # ---- mask inspection file (check 3) -------------------------------------------------------------------------
    lines = []
    cl = meta[meta.kind == "clean"]
    picks = [int(cl.index[cl.relevance_group == g][0]) for g in ("relevant", "nonrelevant") if (cl.relevance_group == g).any()]
    for name in list(dict.fromkeys(atk.attack_name))[:5]:
        ia = int(atk.index[atk.attack_name == name][0])
        picks += [int(ic_of_a[list(atk.index).index(ia)]), ia]
    for r in picks:
        s = ts[int(meta.seq_id[r])]
        dec = lambda m: " ".join(tok.convert_ids_to_tokens(s["input_ids"][m].tolist()))  # noqa: E731
        q, d, ins, am = s["query_mask"], s["doc_mask"], s["inserted_mask"], s["attention_mask"]
        lines += [f"=== seq {meta.seq_id[r]} kind={kinds[r]} group={meta.relevance_group[r]} pair={meta.pair_id[r]} "
                  f"attack={meta.attack_name[r]} successful={meta.successful[r]} n_tokens={len(am)}",
                  f"KEYS full_visible={int(am.sum())}  shared_key={int((am & ~ins).sum())}  "
                  f"max attention on invisible keys (all layers/heads/rows) = {masked_key_max[r]:.3g}",
                  f"QUERY src ({int(q.sum())}): {dec(q)}",
                  f"DOC   src ({int(d.sum())}): {dec(d)}",
                  f"ORIG  src ({int((d & ~ins).sum())}): {dec(d & ~ins)}",
                  f"INS   src ({int((d & ins).sum())}): {dec(d & ins)}",
                  f"MASKED CONTROL SLOTS ({int((ins & ~am).sum())}) at positions {np.where(ins & ~am)[0].tolist()}: {dec(ins & ~am)}",
                  f"TEMPLATE (keys only, never a source region): {dec(~(q | d) & am)}", ""]
    (st / "mask_inspection.txt").write_text("\n".join(lines))

    # ---- save ---------------------------------------------------------------------------------------------------
    sm = meta[KEEP_META].copy()
    sm["score_recomputed"] = score
    toks = [ts[int(i)] for i in meta.seq_id]
    sm["n_keys_full_visible"] = [int(s["attention_mask"].sum()) for s in toks]
    sm["n_keys_shared_key"] = [int((s["attention_mask"] & ~s["inserted_mask"]).sum()) for s in toks]
    for rname, f in (("query", lambda s: s["query_mask"]), ("doc", lambda s: s["doc_mask"]),
                     ("orig", lambda s: s["doc_mask"] & ~s["inserted_mask"]), ("ins", lambda s: s["doc_mask"] & s["inserted_mask"])):
        sm[f"n_tokens_{rname}"] = [int(f(s).sum()) for s in toks]
    sm.to_csv(st / "sequences_meta.csv", index=False)
    np.savez_compressed(st / "attention_statistics.npz", R=R, M=M, seq_id=meta.seq_id.values, regions=np.array(AT.REGIONS),
                        scopes=np.array(AT.SCOPES), metrics=np.array(AT.METRICS), mass_dirs=np.array(AT.MASS_NAMES),
                        mass_kinds=np.array(["raw", "norm"]), heads=np.array(LABELS))
    long = []
    ia_rows = np.where(~na)[0]
    for ri, rname in enumerate(AT.REGIONS):
        for si, sc in enumerate(AT.SCOPES):
            rows = np.arange(n) if (rname in ("query", "doc") and sc == "full_visible") else ia_rows
            if len(rows) == 0:
                continue
            df = pd.DataFrame({"seq_id": np.repeat(meta.seq_id.values[rows], 144), "head_label": np.tile(LABELS, len(rows)),
                               "source_region": rname, "key_scope": sc})
            for mi, m in enumerate(AT.METRICS):
                df[m] = R[rows, ri, si, mi].reshape(-1)
            long.append(df)
    pd.concat(long).to_csv(st / "attention_statistics.csv.gz", index=False, float_format="%.6g", compression="gzip")
    mdf = pd.DataFrame({"seq_id": np.repeat(meta.seq_id.values[ia_rows], 144), "head_label": np.tile(LABELS, len(ia_rows))})
    for di, dname in enumerate(AT.MASS_NAMES):
        mdf[f"{dname}_raw"] = M[ia_rows, di, 0].reshape(-1)
        mdf[f"{dname}_norm"] = M[ia_rows, di, 1].reshape(-1)
    mdf.to_csv(st / "injected_mass.csv.gz", index=False, float_format="%.6g", compression="gzip")
    write_status(st, {"status": "success", "finished": now(), "seconds": time.time() - t0, "n_sequences": n})
    print(json.dumps(val["forward"], indent=1))
    print(f"[01] -> {st} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
