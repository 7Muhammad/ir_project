"""
scripts/18_run_paired_forward.py
=================================
Stage 18 — padded control and attacked input of EVERY aligned instance of the
paired manifest (no success filter here; success is decided from these scores).

ONE full forward per batch (exp16lib.paired.forward_with_heads =
exp16lib.engine.forward_batch(with_score=True) + exp16lib.heads.EncoderHeadCapture
on the same pass) returns, per sequence:
  score                      logit("true") - logit("false"), first decoder step (Exp 01 / Exp 16 definition)
  {side}_ckpt_{checkpoint}   the 25 encoder checkpoint query/document cosines (stage 03 metric)
  {side}_head_{LxHy}         the scope's head cosines, pre-o_proj (stage 13 metric): 36 heads of
                             layers 9-11 (paired.scope L9_L11) or 144 heads of layers 0-11 (all_layers)
Control and attack sequences of one attack are mixed in length-sorted batches.
  delta_score = score_attack - score_control;   successful = delta_score > 0

Read-only cross-check against the ORIGINAL caches on overlapping (attack, qid, docid)
(paired.crosscheck_outputs_dir): Exp 01 cached control/attack scores (atol
validation.fresh_vs_cached_score_atol), stage 03 checkpoint sims and stage 13 attack head
sims (atol validation.head_crosscheck_atol). Missing caches are recorded as skipped.

Resume unit = one attack: 18_paired_forward/per_attack/{attack}/{rows.csv.gz,status.json}
(guarded by the sha256 of the base manifest + that attack's manifest file).
"""

from __future__ import annotations

import json
import pathlib
import sys
import time

import numpy as np
import pandas as pd

EXP_DIR = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EXP_DIR))
import exp16lib  # noqa: E402,F401

from exp16lib import paired as P  # noqa: E402
from exp16lib.checkpoints import CHECKPOINTS  # noqa: E402
from exp16lib.config import load_config, output_dir, resolve_cfg_path  # noqa: E402
from exp16lib.inputs import encode_attack_and_control  # noqa: E402
from exp16lib.run_utils import (file_sha256, is_already_successful, load_model, load_status, now,  # noqa: E402
                                read_jsonl, stage_argparser, unit_is_done, write_status)


def load_paired_manifest(out: pathlib.Path):
    mdir = out / P.MANIFEST_DIR
    if not is_already_successful(mdir, ["base_pairs.jsonl.gz", "fold_map.json"]):
        raise FileNotFoundError(f"run scripts/17_prepare_paired_manifest.py first ({mdir})")
    st = load_status(mdir)
    if file_sha256(mdir / "base_pairs.jsonl.gz") != st["sha256"]["base_pairs"]:
        raise RuntimeError("base_pairs.jsonl.gz changed after stage 17 (sha mismatch)")
    base = {r["pair_id"]: r for r in read_jsonl(mdir / "base_pairs.jsonl.gz")}
    return mdir, st, base


def crosscheck(a: str, df: pd.DataFrame, cfg: dict) -> dict:
    """Compare with the original Exp 01 / Exp 16 caches where (attack, pair) overlap. Raises beyond tolerance."""
    val = cfg["validation"]
    res = {}
    s_path = resolve_cfg_path(cfg, cfg["data"]["exp1_attacks_dir"]) / a / "scores" / "all_scores.csv"
    if s_path.exists():
        s = pd.read_csv(s_path, dtype={"qid": str, "docid": str})
        m = df[["qid", "docid", "score_control", "score_attack"]].merge(s, on=["qid", "docid"], validate="one_to_one")
        if len(m):
            d = float(max((m.score_control - m.control_score).abs().max(), (m.score_attack - m.attack_score).abs().max()))
            if d > float(val["fresh_vs_cached_score_atol"]):
                raise AssertionError(f"{a}: fresh vs Exp 01 cached score diff {d}")
            res["exp01_scores"] = {"n_overlap": len(m), "max_abs_diff": d}
    old = resolve_cfg_path(cfg, cfg["paired"]["crosscheck_outputs_dir"])
    atol = float(val["head_crosscheck_atol"])
    p3 = old / "03_attack" / "per_attack" / a / "rows.csv.gz"
    if p3.exists():
        o = pd.read_csv(p3, dtype={"pair_id": str}, usecols=["pair_id", "checkpoint_index", "sim_attack", "sim_control"])
        o = o[o.pair_id.isin(set(df.pair_id))]
        if len(o):
            wa = o.pivot(index="pair_id", columns="checkpoint_index", values="sim_attack")
            wc = o.pivot(index="pair_id", columns="checkpoint_index", values="sim_control")
            n = df.set_index("pair_id").loc[wa.index]
            d = float(max(np.abs(n[P.ckpt_cols("attack")].values - wa.values).max(),
                          np.abs(n[P.ckpt_cols("control")].values - wc.values).max()))
            if d > atol:
                raise AssertionError(f"{a}: checkpoint sims differ from stage 03 by {d}")
            res["stage03_ckpt_sims"] = {"n_overlap": len(wa), "max_abs_diff": d}
    p13 = old / "13_late_encoder_heads" / "per_attack" / a / "rows.csv.gz"
    shared = [l for l in P.head_labels() if int(l[1:l.index("H")]) in (9, 10, 11)]   # stage 13 = L9-L11 only
    if p13.exists() and shared:
        o = pd.read_csv(p13, dtype={"pair_id": str}, usecols=["pair_id"] + [f"enc_{l}" for l in shared])
        o = o[o.pair_id.isin(set(df.pair_id))].set_index("pair_id")
        if len(o):
            n = df.set_index("pair_id").loc[o.index]
            d = float(np.abs(n[[f"attack_head_{l}" for l in shared]].values - o[[f"enc_{l}" for l in shared]].values).max())
            if d > atol:
                raise AssertionError(f"{a}: head sims differ from stage 13 by {d}")
            res["stage13_head_sims"] = {"n_overlap": len(o), "max_abs_diff": d}
    if P.SCOPE != P.DEFAULT_SCOPE:                  # scope extension: must reproduce the primary L9-L11 run
        pp = output_dir(cfg) / P.FORWARD_DIR[:-len(P.scope_suffix(P.SCOPE))] \
            / "per_attack" / a / "rows.csv.gz"
        if pp.exists():
            cols = ["pair_id", "score_control", "score_attack"] + [f"{sd}_head_{l}" for sd in ("control", "attack")
                                                                   for l in shared]
            o = pd.read_csv(pp, dtype={"pair_id": str}, usecols=cols).set_index("pair_id")
            n = df.set_index("pair_id")
            if set(o.index) != set(n.index):
                raise AssertionError(f"{a}: instance set differs from the primary-scope forward")
            n = n.loc[o.index]
            d = float(np.abs(n[cols[1:]].values - o[cols[1:]].values).max())
            if d > atol:
                raise AssertionError(f"{a}: shared heads/scores differ from the primary-scope run by {d}")
            res["primary_scope_forward"] = {"n_overlap": len(o), "n_shared_heads": len(shared), "max_abs_diff": d}
    return res or {"skipped": "no overlapping cache"}


def main():
    p = stage_argparser("Exp 16 stage 18: paired control/attack forward")
    p.add_argument("--attack-start", type=int, default=0)
    p.add_argument("--attack-end", type=int, default=None)
    args = p.parse_args()
    cfg = load_config(args.config)
    pc = P.validate_cfg(cfg)
    out = output_dir(cfg)
    mdir, mst, base = load_paired_manifest(out)
    stage = out / P.FORWARD_DIR
    hs = P.heads()
    order = list(mst["sha256"]["attacks"])
    selected = order[args.attack_start:args.attack_end]
    unit_sha = {a: f"{mst['sha256']['base_pairs']}:{mst['sha256']['attacks'][a]}" for a in order}
    pending = [a for a in selected if not unit_is_done(stage / "per_attack" / a, ["rows.csv.gz"], unit_sha[a], args.force)]
    print(f"[18] {len(order)} attacks, {len(selected)} selected, {len(pending)} to run; {len(base)} base pairs")
    if pending:
        model, tok, true_id, false_id, device = load_model(cfg)
        max_len = int(cfg["model"]["max_length"])
        bs = int(pc["batch_size"])
        t_stage = time.time()
        for k, a in enumerate(pending):
            t0 = time.time()
            unit = stage / "per_attack" / a
            write_status(unit, {"status": "running", "started": now()})
            path = mdir / "attacks" / f"{a}.jsonl.gz"
            if file_sha256(path) != mst["sha256"]["attacks"][a]:
                raise RuntimeError(f"{path} changed after stage 17")
            recs = read_jsonl(path)
            a_seqs, c_seqs = [], []
            for r in recs:
                b = base[r["pair_id"]]
                atk, ctl, info = encode_attack_and_control(tok, b["query"], b["passage"], r["attacked_passage"], max_len)
                if (info["seq_len"], info["n_inserted"], atk.n_doc_tokens, ctl.n_doc_tokens) != \
                        (r["seq_len"], r["n_inserted"], r["n_document_tokens_attack"], r["n_document_tokens_control"]):
                    raise RuntimeError(f"{a} {r['pair_id']}: rebuilt encoding differs from the stage-17 manifest")
                a_seqs.append(atk)
                c_seqs.append(ctl)
            C, H, S = P.run_paired_sequences(model, a_seqs + c_seqs, tok.pad_token_id, device, bs, hs, true_id, false_id)
            n = len(recs)
            df = pd.DataFrame([{m: r[m] for m in P.INSTANCE_META} for r in recs])
            df["score_control"], df["score_attack"] = S[n:], S[:n]
            df["delta_score"] = df.score_attack - df.score_control
            df["successful"] = df.delta_score > 0
            ck = pd.DataFrame(np.hstack([C[n:], C[:n], H[n:], H[:n]]),
                              columns=P.ckpt_cols("control") + P.ckpt_cols("attack") + P.head_cols("control") + P.head_cols("attack"))
            df = pd.concat([df, ck], axis=1)
            P.check_forward_frame(df)
            checks = crosscheck(a, df, cfg)
            out_df = df.copy()                        # scores exact (repr round-trips) so delta == attack - control
            for c in ("score_control", "score_attack", "delta_score"):
                out_df[c] = [repr(float(v)) for v in df[c]]
            out_df.to_csv(unit / "rows.csv.gz", index=False, float_format="%.9g")
            write_status(unit, {"status": "success", "finished": now(), "manifest_sha256": unit_sha[a],
                                "n_instances": n, "n_successful": int(df.successful.sum()),
                                "n_successful_relevant": int((df.successful & (df.relevance_group == P.RELEVANT)).sum()),
                                "n_successful_nonrelevant": int((df.successful & (df.relevance_group == P.NONRELEVANT)).sum()),
                                "crosscheck": checks, "seconds": time.time() - t0, "device": str(device)})
            el = time.time() - t_stage
            print(f"[18] {a:24s} {n} inst, succ {int(df.successful.sum())}, {time.time() - t0:.1f}s "
                  f"({k + 1}/{len(pending)}, ETA {el / (k + 1) * (len(pending) - k - 1) / 60:.1f} min) "
                  f"xcheck {json.dumps(checks)}", flush=True)
    missing = [a for a in order if not unit_is_done(stage / "per_attack" / a, ["rows.csv.gz"], unit_sha[a], False)]
    if missing:
        print(f"[18] {len(missing)} attacks still missing; stage not marked complete")
        return
    per = []
    for a in order:
        s = load_status(stage / "per_attack" / a)
        per.append({"attack_name": a, **{k: s[k] for k in ("n_instances", "n_successful", "n_successful_relevant",
                                                           "n_successful_nonrelevant", "seconds")},
                    "crosscheck": json.dumps(s["crosscheck"])})
    pd.DataFrame(per).to_csv(stage / "per_attack_status.csv", index=False)
    write_status(stage, {"status": "success", "finished": now(), "n_attacks": len(order),
                         "n_instances": int(sum(r["n_instances"] for r in per)), "checkpoints": CHECKPOINTS,
                         "heads": P.head_labels()})
    print(f"[18] all {len(order)} attacks complete -> {stage}")


if __name__ == "__main__":
    main()
