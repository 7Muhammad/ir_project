"""
scripts/03_run_attack_similarity.py
====================================
Stage 03 — attacked vs matched padded-control similarities for ALL scored
examples of ALL attacks (no success filter).

  attack doc pool  = passage + every injected token (all active)
  control doc pool = active document positions only (insertion slots masked)
  delta_sim(c)     = sim_attack(c) - sim_control(c)
  delta_score      = cached Exp 01 attack_score - control_score

Attack and control sequences of one attack are run together in
length-sorted batches (one forward per batch gives all 25 checkpoints).

Resume unit = one attack: 03_attack/per_attack/{attack}/rows.csv.gz +
status.json (manifest sha). `--attack-start/--attack-end` select a
deterministic slice of the grid order for splitting across SLURM jobs; the
combined 03_attack/attack_similarity.csv.gz is written only when every
attack is complete.

Smoke (validation.fresh_score_check_n > 0): the first n examples of every
attack are freshly scored with Exp 01's own functions and must match the
cached scores.
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
from exp16lib.engine import run_sequences  # noqa: E402
from exp16lib.inputs import encode_attack_and_control  # noqa: E402
from exp16lib.run_utils import load_manifest, load_model, now, stage_argparser, unit_is_done, write_status  # noqa: E402
from exp16lib.tables import long_table  # noqa: E402
from src.model_utils import build_padded_control_and_attack_encodings_general, score_from_encoding  # noqa: E402

META = ["attack_id", "attack_name", "attack_token", "attack_position", "repetitions", "pair_id", "qid", "docid",
        "score_attack", "score_control", "delta_score", "n_query_tokens", "n_document_tokens_attack",
        "n_document_tokens_control", "n_inserted", "alignment_boundary_shift"]


def fresh_score_check(model, tok, recs, n, true_id, false_id, device, max_len, atol):
    worst = 0.0
    for r in recs[:n]:
        c, a, al = build_padded_control_and_attack_encodings_general(
            tok, r["query"], r["passage"], r["attacked_passage"], max_len, device)
        sc = score_from_encoding(model, c, true_id, false_id, device)
        sa = score_from_encoding(model, a, true_id, false_id, device)
        worst = max(worst, abs(sc - r["score_control"]), abs(sa - r["score_attack"]))
    if worst > atol:
        raise AssertionError(f"{recs[0]['attack_name']}: fresh vs cached score diff {worst} > {atol}")
    return worst


def main():
    p = stage_argparser("Exp 16 stage 03: attack vs control similarity")
    p.add_argument("--attack-start", type=int, default=0)
    p.add_argument("--attack-end", type=int, default=None)
    args = p.parse_args()
    cfg = load_config(args.config)
    out = output_dir(cfg)
    recs, sha = load_manifest(out, "attacks")
    stage = out / "03_attack"
    order = list(dict.fromkeys(r["attack_name"] for r in recs))
    selected = order[args.attack_start:args.attack_end]
    pending = [a for a in selected if not unit_is_done(stage / "per_attack" / a, ["rows.csv.gz"], sha, args.force)]
    print(f"[03] {len(order)} attacks in manifest, {len(selected)} selected, {len(selected) - len(pending)} done, "
          f"{len(pending)} to run")
    if pending:
        model, tok, true_id, false_id, device = load_model(cfg)
        max_len = int(cfg["model"]["max_length"])
        val = cfg["validation"]
        t_stage = time.time()
        for k, attack in enumerate(pending):
            t0 = time.time()
            unit = stage / "per_attack" / attack
            write_status(unit, {"status": "running", "started": now()})
            ra = [r for r in recs if r["attack_name"] == attack]
            a_seqs, c_seqs = [], []
            for r in ra:
                a, c, info = encode_attack_and_control(tok, r["query"], r["passage"], r["attacked_passage"], max_len)
                if (info["seq_len"], info["n_inserted"], a.n_doc_tokens, c.n_doc_tokens) != \
                        (r["seq_len"], r["n_inserted"], r["n_document_tokens_attack"], r["n_document_tokens_control"]):
                    raise RuntimeError(f"{attack} {r['pair_id']}: rebuilt encoding differs from manifest")
                a_seqs.append(a)
                c_seqs.append(c)
            sims, _ = run_sequences(model, a_seqs + c_seqs, tok.pad_token_id, device, int(cfg["runtime"]["batch_size"]))
            n = len(ra)
            s_att, s_ctl = sims[:n], sims[n:]
            df = long_table([{k2: r[k2] for k2 in META} for r in ra],
                            {"sim_attack": s_att, "sim_control": s_ctl, "delta_sim": s_att - s_ctl})
            df.to_csv(unit / "rows.csv.gz", index=False)
            st = {"status": "success", "finished": now(), "manifest_sha256": sha, "n_examples": n,
                  "n_rows": len(df), "seconds": time.time() - t0, "device": str(device)}
            n_fresh = int(val.get("fresh_score_check_n", 0))
            if n_fresh:
                st["fresh_vs_cached_score_max_abs_diff"] = fresh_score_check(
                    model, tok, ra, n_fresh, true_id, false_id, device, max_len, float(val["fresh_vs_cached_score_atol"]))
            write_status(unit, st)
            el = time.time() - t_stage
            print(f"[03] {attack:24s} {n} ex, {time.time() - t0:.1f}s ({k + 1}/{len(pending)}, "
                  f"elapsed {el / 60:.1f} min, ETA {el / (k + 1) * (len(pending) - k - 1) / 60:.1f} min)", flush=True)
    missing = [a for a in order if not unit_is_done(stage / "per_attack" / a, ["rows.csv.gz"], sha, False)]
    if missing:
        print(f"[03] {len(missing)} attacks still missing; combined table not written yet")
        return
    df = pd.concat([pd.read_csv(stage / "per_attack" / a / "rows.csv.gz", dtype={"qid": str, "docid": str, "attack_token": str, "attack_name": str, "pair_id": str})
                    for a in order], ignore_index=True)
    if not np.allclose(df["delta_sim"], df["sim_attack"] - df["sim_control"], atol=1e-12):
        raise RuntimeError("delta_sim != sim_attack - sim_control")
    df.to_csv(stage / "attack_similarity.csv.gz", index=False)
    write_status(stage, {"status": "success", "finished": now(), "manifest_sha256": sha,
                         "n_attacks": len(order), "n_rows": len(df)})
    print(f"[03] combined {len(order)} attacks, {len(df)} rows -> {stage / 'attack_similarity.csv.gz'}")


if __name__ == "__main__":
    main()
