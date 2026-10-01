"""
scripts/17_prepare_paired_manifest.py
======================================
Stage 17 — manifest of the PAIRED extension (tokenizer only, no model).

Population: every judged attackable DL19 base pair (qrel 2/3 or 0; qrel 1 and
unjudged dropped via exp16lib.qrels.relevance_group) x every attack of the
105-attack grid (exp16lib.config.get_attacks = Exp 01 registry +
multi_attack.yaml), read directly from the upstream injected TSVs with Exp 01's
own reader (src.data_utils.load_attacked_tsv; text = attacked, text_0 = clean).
Exp 01 selected_examples.jsonl (success-filtered) is NOT used.

Checks (fail loudly):
  * no duplicate (qid, docno) inside an attack file;
  * every attack file has the same base set {(qid, docno) -> (query, text_0)};
  * every instance is encoded with exp16lib.inputs.encode_attack_and_control;
    Exp 01 alignment failures are LOGGED (alignment_failures.csv) and excluded,
    any other EncodingError is raised (the stage-00 rule).

Outputs (17_paired_manifest/):
  base_pairs.jsonl.gz                 judged base pairs (query, passage, grade, group, rank)
  attacks/{attack}.jsonl.gz           aligned instances of one attack (attacked text + layout)
  alignment_failures.csv              every excluded instance with the reason
  attack_population_summary.csv       per attack counts (by group)
  fold_map.json                       qid -> fold (query_folds over ALL judged attackable qids, 5, seed 42)
  old_outputs_fingerprint.json        sha256 of every pre-existing Exp 16 output (checked again in stage 21)
  provenance.json, status.json        (status holds the sha256 of every manifest file)
"""

from __future__ import annotations

import csv
import json
import pathlib
import sys
import time
from collections import Counter

import pandas as pd

EXP_DIR = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EXP_DIR))
import exp16lib  # noqa: E402,F401

from exp16lib import paired as P  # noqa: E402
from exp16lib.anomaly import query_folds  # noqa: E402
from exp16lib.config import get_attacks, load_config, output_dir  # noqa: E402
from exp16lib.qrels import relevance_group  # noqa: E402
from exp16lib.run_utils import (file_sha256, is_already_successful, load_tokenizer, now, stage_argparser,  # noqa: E402
                                write_jsonl, write_status)
from src.data_utils import load_attacked_tsv  # noqa: E402

csv.field_size_limit(sys.maxsize)
BIG = 10 ** 9


def load_qrels(cfg):
    import ir_datasets
    ds = ir_datasets.load(cfg["qrels"]["irds_qrels_id"])
    q = {}
    for r in ds.qrels_iter():
        key = (str(r.query_id), str(r.doc_id))
        if key in q:
            raise RuntimeError(f"duplicate qrel {key}")
        q[key] = int(r.relevance)
    return q, ir_datasets.__version__


def main():
    args = stage_argparser("Exp 16 stage 17: paired manifest").parse_args()
    cfg = load_config(args.config)
    pc = P.validate_cfg(cfg)
    out = output_dir(cfg)
    mdir = out / P.MANIFEST_DIR
    if not args.force and is_already_successful(mdir, ["base_pairs.jsonl.gz", "fold_map.json", "provenance.json"]):
        print(f"[17] paired manifest already complete in {mdir} (use --force to rebuild)")
        return
    t0 = time.time()
    write_status(mdir, {"status": "running", "started": now()})
    # Fingerprint the ORIGINAL Exp 16 outputs before anything paired is written.
    fp_path = mdir / "old_outputs_fingerprint.json"
    if not fp_path.exists() or args.force:
        fp = P.fingerprint_old_outputs(out)
        fp_path.write_text(json.dumps({"created": now(), "n_files": len(fp), "sha256": fp}, indent=1))
        print(f"[17] fingerprinted {len(fp)} pre-existing Exp 16 output files")

    tok = load_tokenizer(cfg)
    max_len = int(cfg["model"]["max_length"])
    specs = get_attacks(cfg)
    qrels, irds_version = load_qrels(cfg)

    # ---- base set (identical across every attack file) --------------------------------
    base, tsv_rows = None, {}
    for spec in specs:
        rows = load_attacked_tsv(spec.path, BIG, BIG)
        keys = [(r["qid"], r["docid"]) for r in rows]
        if len(set(keys)) != len(keys):
            raise RuntimeError(f"{spec.attack_name}: duplicate (qid, docno) rows")
        b = {(r["qid"], r["docid"]): (r["query"], r["passage"], r["rank"]) for r in rows}
        if base is None:
            base = b
        elif {k: v[:2] for k, v in b.items()} != {k: v[:2] for k, v in base.items()}:
            raise RuntimeError(f"{spec.attack_name}: base (qid, docno, query, text_0) set differs from {specs[0].attack_name}")
        tsv_rows[spec.attack_name] = {(r["qid"], r["docid"]): r["attacked_passage"] for r in rows   # judged only
                                      if qrels.get((r["qid"], r["docid"])) in (0, 2, 3)}

    grades = Counter()
    judged = {}
    for key, (query, passage, rank) in base.items():
        g = qrels.get(key)
        grp = None if g is None else relevance_group(g)
        grades["unjudged" if g is None else f"grade_{g}"] += 1
        if grp is not None:
            judged[key] = {"pair_id": P.pair_id(*key), "qid": key[0], "docid": key[1], "query": query,
                           "passage": passage, "bm25_rank": rank, "qrel_grade": g, "relevance_group": grp}
    all_judged_qids = sorted({k[0] for k in judged})

    # ---- smoke-only trimming (deterministic) -----------------------------------------
    smoke = pc.get("max_queries") is not None or pc.get("max_pairs_per_group_per_query") is not None
    if smoke:
        with_rel = sorted({k[0] for k, v in judged.items() if v["relevance_group"] == P.RELEVANT}, key=lambda x: (len(x), x))
        keep_q = set(with_rel[:int(pc["max_queries"])] if pc.get("max_queries") is not None else with_rel)
        kmax = pc.get("max_pairs_per_group_per_query")
        kept = {}
        for qid in sorted(keep_q):
            for grp in P.GROUPS:
                ks = sorted((k for k, v in judged.items() if k[0] == qid and v["relevance_group"] == grp),
                            key=lambda k: (judged[k]["bm25_rank"], k[1]))   # best BM25 rank (overlaps Exp 01 caches)
                for k in (ks if kmax is None else ks[:int(kmax)]):
                    kept[k] = judged[k]
        judged = kept
    base_recs = [judged[k] for k in sorted(judged, key=lambda k: (len(k[0]), k[0], k[1]))]
    qids = sorted({r["qid"] for r in base_recs})
    fold_map = query_folds(qids, P.N_FOLDS, P.FOLD_SEED)

    # ---- encode every instance; log alignment failures ---------------------------------
    (mdir / "attacks").mkdir(parents=True, exist_ok=True)
    failures, per_attack, sha = [], [], {}
    for attack_id, spec in enumerate(specs):
        a = spec.attack_name
        recs, n_fail = [], Counter()
        for b in base_recs:
            key = (b["qid"], b["docid"])
            attacked = tsv_rows[a][key]
            atk, ctl, info = P.encode_instance(tok, b["query"], b["passage"], attacked, max_len)
            if atk is None:                                   # Exp 01 alignment failure: log + exclude
                failures.append({"attack_id": attack_id, "attack_name": a, "pair_id": b["pair_id"], "qid": b["qid"],
                                 "docid": b["docid"], "qrel_grade": b["qrel_grade"],
                                 "relevance_group": b["relevance_group"], "reason": info})
                n_fail[b["relevance_group"]] += 1
                continue
            recs.append({"attack_id": attack_id, "attack_name": a, "attack_token": spec.token,
                         "attack_position": spec.position, "repetitions": spec.repetitions,
                         "pair_id": b["pair_id"], "qid": b["qid"], "docid": b["docid"],
                         "qrel_grade": b["qrel_grade"], "relevance_group": b["relevance_group"],
                         "attacked_passage": attacked, "seq_len": info["seq_len"], "n_inserted": info["n_inserted"],
                         "n_query_tokens": atk.n_query_tokens, "n_document_tokens_attack": atk.n_doc_tokens,
                         "n_document_tokens_control": ctl.n_doc_tokens,
                         "alignment_boundary_shift": info["alignment_boundary_shift"]})
        path = mdir / "attacks" / f"{a}.jsonl.gz"
        write_jsonl(path, recs)
        sha[a] = file_sha256(path)
        n_by = Counter(r["relevance_group"] for r in recs)
        per_attack.append({"attack_id": attack_id, "attack_name": a, "token": spec.token, "position": spec.position,
                           "repetitions": spec.repetitions, "tsv": str(spec.path),
                           "n_base_pairs": len(base_recs), "n_aligned": len(recs),
                           "n_align_failed": sum(n_fail.values()),
                           "n_aligned_relevant": n_by[P.RELEVANT], "n_aligned_nonrelevant": n_by[P.NONRELEVANT],
                           "n_align_failed_relevant": n_fail[P.RELEVANT],
                           "n_align_failed_nonrelevant": n_fail[P.NONRELEVANT],
                           "n_boundary_shift": sum(r["alignment_boundary_shift"] for r in recs)})
        print(f"[17] {a:24s} aligned={len(recs)} failed={sum(n_fail.values())} "
              f"({attack_id + 1}/{len(specs)}, {time.time() - t0:.0f}s)", flush=True)

    write_jsonl(mdir / "base_pairs.jsonl.gz", base_recs)
    pd.DataFrame(failures, columns=["attack_id", "attack_name", "pair_id", "qid", "docid", "qrel_grade",
                                    "relevance_group", "reason"]).to_csv(mdir / "alignment_failures.csv", index=False)
    pa = pd.DataFrame(per_attack)
    pa.to_csv(mdir / "attack_population_summary.csv", index=False)
    (mdir / "fold_map.json").write_text(json.dumps({"n_folds": P.N_FOLDS, "seed": P.FOLD_SEED,
                                                    "qids": qids, "fold_of": fold_map}, indent=1))
    n_rel = sum(r["relevance_group"] == P.RELEVANT for r in base_recs)
    prov = {
        "config": cfg["_config_path"], "attack_grid_source": cfg["_attack_grid_source"], "created": now(),
        "tsv_dir": str(specs[0].path.parent), "tsv_reader": "src.data_utils.load_attacked_tsv (Exp 01)",
        "n_attacks": len(specs), "attack_names": [s.attack_name for s in specs],
        "qrels": {"irds_qrels_id": cfg["qrels"]["irds_qrels_id"], "ir_datasets_version": irds_version,
                  "groups": {"relevant": [2, 3], "nonrelevant": [0], "excluded": [1, "unjudged"]}},
        "base_pairs_in_tsv": len(base), "base_pair_grade_counts": dict(sorted(grades.items())),
        "n_judged_attackable_qids_full": len(all_judged_qids),
        "smoke_trimmed": smoke,
        "n_base_pairs": len(base_recs), "n_base_relevant": n_rel, "n_base_nonrelevant": len(base_recs) - n_rel,
        "n_queries": len(qids), "fold_sizes_queries": pd.Series(fold_map).value_counts().sort_index().tolist(),
        "n_instances_before_alignment": len(base_recs) * len(specs),
        "n_instances_before_alignment_relevant": n_rel * len(specs),
        "n_instances_before_alignment_nonrelevant": (len(base_recs) - n_rel) * len(specs),
        "n_alignment_failures": len(failures),
        "n_alignment_failures_relevant": int(pa.n_align_failed_relevant.sum()),
        "n_alignment_failures_nonrelevant": int(pa.n_align_failed_nonrelevant.sum()),
        "n_attacks_with_failures": int((pa.n_align_failed > 0).sum()),
        "n_instances_aligned": int(pa.n_aligned.sum()),
        "n_boundary_shift": int(pa.n_boundary_shift.sum()),
        "seconds": time.time() - t0,
    }
    (mdir / "provenance.json").write_text(json.dumps(prov, indent=2))
    write_status(mdir, {"status": "success", "finished": now(),
                        "sha256": {"base_pairs": file_sha256(mdir / "base_pairs.jsonl.gz"), "attacks": sha},
                        "n_instances_aligned": prov["n_instances_aligned"], "n_alignment_failures": len(failures)})
    print(json.dumps({k: prov[k] for k in ("base_pairs_in_tsv", "base_pair_grade_counts", "n_base_relevant",
                                           "n_base_nonrelevant", "n_queries", "n_instances_before_alignment",
                                           "n_alignment_failures", "n_instances_aligned")}, indent=1))
    print(f"[17] -> {mdir}")


if __name__ == "__main__":
    main()
