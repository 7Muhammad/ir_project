"""
scripts/02_run_qrel_similarity.py
==================================
Stage 02 — similarities for the balanced within-query DL19 qrel sample
(qrel 2/3 "relevant" vs qrel 0 "nonrelevant"; human labels define the class).

The same forward pass also yields the monoT5 score of each judged document
(one decoder step; stored as `monot5_score`, descriptive only — never used to
select or label documents).

Resume unit = one query: 02_qrel/per_query/{qid}/rows.csv.gz + status.json
(manifest sha). When all queries are done: 02_qrel/qrel_similarity.csv.gz.
"""

from __future__ import annotations

import pathlib
import sys

import pandas as pd

EXP_DIR = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EXP_DIR))
import exp16lib  # noqa: E402,F401

from exp16lib.config import load_config, output_dir  # noqa: E402
from exp16lib.engine import run_sequences  # noqa: E402
from exp16lib.inputs import encode_clean  # noqa: E402
from exp16lib.run_utils import load_manifest, load_model, now, stage_argparser, unit_is_done, write_status  # noqa: E402
from exp16lib.tables import long_table  # noqa: E402

META = ["qid", "docid", "pair_id", "query", "qrel_grade", "relevance_group", "in_bm25_top1000"]


def main():
    args = stage_argparser("Exp 16 stage 02: qrel similarity").parse_args()
    cfg = load_config(args.config)
    out = output_dir(cfg)
    recs, sha = load_manifest(out, "qrel")
    stage = out / "02_qrel"
    qids = list(dict.fromkeys(r["qid"] for r in recs))
    pending = [q for q in qids if not unit_is_done(stage / "per_query" / q, ["rows.csv.gz"], sha, args.force)]
    print(f"[02] {len(qids)} queries, {len(qids) - len(pending)} done, {len(pending)} to run")
    if pending:
        model, tok, true_id, false_id, device = load_model(cfg)
        max_len = int(cfg["model"]["max_length"])
        for q in pending:
            unit = stage / "per_query" / q
            write_status(unit, {"status": "running", "started": now()})
            rq = [r for r in recs if r["qid"] == q]
            seqs = [encode_clean(tok, r["query"], r["passage"], max_len) for r in rq]
            for r, s in zip(rq, seqs):
                if (s.seq_len, s.n_query_tokens, s.n_doc_tokens) != (r["seq_len"], r["n_query_tokens"], r["n_document_tokens"]):
                    raise RuntimeError(f"{r['pair_id']}: rebuilt encoding differs from manifest")
            sims, scores = run_sequences(model, seqs, tok.pad_token_id, device, int(cfg["runtime"]["batch_size"]),
                                         with_score=True, true_id=true_id, false_id=false_id)
            meta = [{k: r[k] for k in META} | {"monot5_score": float(sc), "n_query_tokens": s.n_query_tokens,
                                              "n_document_tokens": s.n_doc_tokens}
                    for r, s, sc in zip(rq, seqs, scores)]
            long_table(meta, {"similarity": sims}).to_csv(unit / "rows.csv.gz", index=False)
            write_status(unit, {"status": "success", "finished": now(), "manifest_sha256": sha,
                                "n_docs": len(rq), "device": str(device)})
            print(f"[02] qid {q}: {len(rq)} docs", flush=True)
    df = pd.concat([pd.read_csv(stage / "per_query" / q / "rows.csv.gz", dtype={"qid": str, "docid": str})
                    for q in qids], ignore_index=True)
    df.to_csv(stage / "qrel_similarity.csv.gz", index=False)
    write_status(stage, {"status": "success", "finished": now(), "manifest_sha256": sha,
                         "n_queries": len(qids), "n_rows": len(df)})
    print(f"[02] {len(recs)} judged docs over {len(qids)} queries -> {stage / 'qrel_similarity.csv.gz'}")


if __name__ == "__main__":
    main()
