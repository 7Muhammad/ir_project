"""
scripts/00_prepare_manifests.py
================================
Stage 00 — build the three immutable populations + provenance (tokenizer
only, no model):

  manifests/clean.jsonl            canonical 500 Type-A pairs + cached clean score
  manifests/attacks.jsonl.gz       ALL scored examples x ALL attacks (no success filter)
                                   + cached control/attack scores
  manifests/qrel_balanced.jsonl    balanced within-query DL19 qrel sample (2/3 vs 0, seed 42)
  manifests/qrel_query_summary.csv per-query counts (incl. grade-1 exclusions)
  manifests/qrel_doc_texts.jsonl.gz passage text of every judged docid (ir_datasets)
  manifests/provenance.json        exact sources, sha256s, counts, cross-checks

Population checks (fail loudly):
  * every attack's pairs.jsonl == the canonical 500 pairs (query + passage);
  * per attack: the pairs whose Exp 01 alignment fails are EXACTLY the pairs
    missing from all_scores.csv, so all_scores.csv is the complete aligned
    population (and selected_examples.jsonl is not used);
  * every manifest example encodes with valid query/doc masks, no truncation.
"""

from __future__ import annotations

import csv
import gzip
import json
import pathlib
import sys
from collections import Counter

import pandas as pd

EXP_DIR = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EXP_DIR))
import exp16lib  # noqa: E402,F401

from exp16lib.config import get_attacks, load_config, output_dir, resolve_cfg_path  # noqa: E402
from exp16lib.inputs import EncodingError, encode_attack_and_control, encode_clean  # noqa: E402
from exp16lib.qrels import build_balanced_manifest  # noqa: E402
from exp16lib.run_utils import (file_sha256, is_already_successful, load_tokenizer, now,  # noqa: E402
                                read_jsonl, stage_argparser, write_jsonl, write_status)

csv.field_size_limit(sys.maxsize)


def pair_id(qid, docid) -> str:
    return f"{qid}_{docid}"


def build_clean(cfg, tok, prov):
    pairs_path = resolve_cfg_path(cfg, cfg["data"]["canonical_pairs"])
    scores_path = resolve_cfg_path(cfg, cfg["data"]["canonical_scores"])
    pairs = read_jsonl(pairs_path)
    scores = pd.read_csv(scores_path, dtype={"qid": str, "docid": str}).set_index(["qid", "docid"])
    if len(pairs) != 500 or len({(p["qid"], p["docid"]) for p in pairs}) != 500:
        raise RuntimeError(f"expected 500 unique canonical pairs in {pairs_path}")
    max_len = int(cfg["model"]["max_length"])
    recs = []
    for p in pairs:
        qid, docid = str(p["qid"]), str(p["docid"])
        seq = encode_clean(tok, p["query"], p["passage"], max_len)
        recs.append({
            "pair_id": pair_id(qid, docid), "qid": qid, "docid": docid, "rank": p.get("rank"),
            "query": p["query"], "passage": p["passage"],
            "score_clean": float(scores.loc[(qid, docid), "original_score"]),
            "seq_len": seq.seq_len, "n_query_tokens": seq.n_query_tokens, "n_document_tokens": seq.n_doc_tokens,
        })
    prov["clean"] = {
        "canonical_pairs": str(pairs_path), "canonical_pairs_sha256": file_sha256(pairs_path),
        "clean_scores": str(scores_path), "clean_scores_sha256": file_sha256(scores_path),
        "score_column": "original_score (Exp 01 Type-A score, logit(true)-logit(false))",
        "n_pairs": len(recs), "n_queries": len({r["qid"] for r in recs}),
    }
    return recs, {(r["qid"], r["docid"]): r for r in recs}


def build_attacks(cfg, tok, canonical, prov):
    exp1 = resolve_cfg_path(cfg, cfg["data"]["exp1_attacks_dir"])
    specs = get_attacks(cfg)
    max_len = int(cfg["model"]["max_length"])
    cap = cfg["data"].get("max_examples_per_attack")
    recs, per_attack = [], []
    for attack_id, spec in enumerate(specs):
        a = spec.attack_name
        pairs = read_jsonl(exp1 / a / "pairs" / "pairs.jsonl")
        scores = pd.read_csv(exp1 / a / "scores" / "all_scores.csv", dtype={"qid": str, "docid": str})
        status = json.loads((exp1 / a / "status.json").read_text())
        if status.get("status") != "success":
            raise RuntimeError(f"Exp 01 attack {a} did not complete: {status}")
        if scores[["qid", "docid"]].duplicated().any():
            raise RuntimeError(f"{a}: duplicate (qid, docid) in all_scores.csv")
        score_idx = scores.set_index(["qid", "docid"])
        by_key = {}
        failed = set()
        for p in pairs:
            key = (str(p["qid"]), str(p["docid"]))
            c = canonical.get(key)
            if c is None or c["query"] != p["query"] or c["passage"] != p["passage"]:
                raise RuntimeError(f"{a}: pair {key} differs from the canonical 500 pairs")
            try:
                atk, ctl, info = encode_attack_and_control(tok, p["query"], p["passage"], p["attacked_passage"], max_len)
            except EncodingError as e:
                if "alignment failed" not in str(e):
                    raise
                failed.add(key)
                continue
            by_key[key] = (p, atk, ctl, info)
        scored = set(score_idx.index)
        if scored != set(by_key) or failed & scored:
            raise RuntimeError(f"{a}: aligned pairs ({len(by_key)}) != scored pairs in all_scores.csv ({len(scored)})")
        if len(scored) + len(failed) != len(pairs) or int(status.get("n_align_failed", 0)) != len(failed):
            raise RuntimeError(f"{a}: population accounting mismatch (scored {len(scored)}, failed {len(failed)})")
        # canonical pair order, then optional smoke cap
        keys = [(str(p["qid"]), str(p["docid"])) for p in pairs if (str(p["qid"]), str(p["docid"])) in by_key]
        if cap is not None:
            keys = keys[:int(cap)]
        deltas = []
        for key in keys:
            p, atk, ctl, info = by_key[key]
            s = score_idx.loc[key]
            if abs(float(s["original_score"]) - canonical[key]["score_clean"]) > 1e-9:
                raise RuntimeError(f"{a} {key}: original_score differs from canonical clean score")
            d = float(s["attack_score"]) - float(s["control_score"])
            deltas.append(d)
            recs.append({
                "attack_id": attack_id, "attack_name": a, "attack_token": spec.token,
                "attack_position": spec.position, "repetitions": spec.repetitions,
                "pair_id": pair_id(*key), "qid": key[0], "docid": key[1],
                "query": p["query"], "passage": p["passage"], "attacked_passage": p["attacked_passage"],
                "score_control": float(s["control_score"]), "score_attack": float(s["attack_score"]),
                "delta_score": d, "score_clean": canonical[key]["score_clean"],
                "seq_len": info["seq_len"], "n_inserted": info["n_inserted"],
                "n_query_tokens": atk.n_query_tokens, "n_document_tokens_attack": atk.n_doc_tokens,
                "n_document_tokens_control": ctl.n_doc_tokens,
                "alignment_boundary_shift": info["alignment_boundary_shift"],
            })
        per_attack.append({
            "attack_id": attack_id, "attack_name": a, "token": spec.token, "position": spec.position,
            "repetitions": spec.repetitions, "tsv": str(spec.path),
            "n_pairs": len(pairs), "n_align_failed": len(failed), "n_scored": len(scored), "n_used": len(keys),
            "n_delta_pos": sum(d > 0 for d in deltas), "n_delta_zero": sum(d == 0 for d in deltas),
            "n_delta_neg": sum(d < 0 for d in deltas),
            "all_scores_sha256": file_sha256(exp1 / a / "scores" / "all_scores.csv"),
        })
        print(f"[00] {a:24s} pairs={len(pairs)} failed_align={len(failed)} used={len(keys)}", flush=True)
    prov["attacks"] = {
        "attack_config": cfg["_attack_grid_source"], "attack_names": [s.attack_name for s in specs],
        "n_attacks": len(specs),
        "exp1_attacks_dir": str(exp1),
        "population_source": "{attack}/scores/all_scores.csv (every aligned pair; NOT selected_examples.jsonl)",
        "text_source": "{attack}/pairs/pairs.jsonl (attacked_passage from the ECIR-24 injected TSVs)",
        "success_filter": "none", "max_examples_per_attack": cap,
        "n_examples_total": len(recs),
        "n_boundary_shift": sum(r["alignment_boundary_shift"] for r in recs),
        "per_attack": per_attack,
    }
    return recs


def build_qrels(cfg, tok, prov, out):
    import ir_datasets
    qc = cfg["qrels"]
    ds = ir_datasets.load(qc["irds_qrels_id"])
    qrels = [(q.query_id, q.doc_id, int(q.relevance)) for q in ds.qrels_iter()]
    queries = {q.query_id: q.text for q in ds.queries_iter()}
    qrels_path = pathlib.Path(ir_datasets.util.home_path()) / "msmarco-passage" / "trec-dl-2019" / "qrels"
    needed = {d for _, d, _ in qrels}

    # Passage text for every judged docid (cached across runs of this stage).
    text_cache = out / "manifests" / "qrel_doc_texts.jsonl.gz"
    texts = {}
    if text_cache.exists():
        texts = {r["docid"]: r["text"] for r in read_jsonl(text_cache)}
    if not needed <= set(texts):
        print(f"[00] streaming {qc['irds_docs_id']} for {len(needed)} judged docids ...", flush=True)
        texts = {}
        for d in ir_datasets.load(qc["irds_docs_id"]).docs_iter():
            if d.doc_id in needed:
                texts[d.doc_id] = d.text
                if len(texts) == len(needed):
                    break
        write_jsonl(text_cache, [{"docid": k, "text": v} for k, v in sorted(texts.items())])
    missing = needed - set(texts)
    if missing:
        raise RuntimeError(f"{len(missing)} judged docids have no text in {qc['irds_docs_id']}")

    # Cross-check against the ECIR-24 BM25 run text (same queries / passages).
    bm_q, bm_t = {}, {}
    with gzip.open(cfg["data"]["bm25_run_text"], "rt", encoding="utf-8") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            bm_q[row["qid"]] = row["query"]
            bm_t[row["docno"]] = row["text"]
    overlap = [d for d in needed if d in bm_t]
    same_text = sum(bm_t[d] == texts[d] for d in overlap)
    judged_qids = sorted({q for q, _, _ in qrels})
    q_same = sum(bm_q.get(q) == queries[q] for q in judged_qids)

    # Eligibility (decided BEFORE balancing): prompt must fit max_length.
    max_len = int(cfg["model"]["max_length"])
    eligible, too_long = set(), []
    for qid, docid, _ in qrels:
        try:
            encode_clean(tok, queries[qid], texts[docid], max_len)
            eligible.add((qid, docid))
        except EncodingError as e:
            too_long.append({"qid": qid, "docid": docid, "reason": str(e)})
    records, summary = build_balanced_manifest(qrels, int(qc["seed"]), eligible)

    # Smoke-only trimming (after balancing; keeps the classes equal).
    if qc.get("max_queries") is not None or qc.get("max_docs_per_class") is not None:
        keep_q = [s["qid"] for s in summary if s["retained"]]
        if qc.get("max_queries") is not None:
            keep_q = keep_q[:int(qc["max_queries"])]
        k = qc.get("max_docs_per_class")
        trimmed = []
        for qid in keep_q:
            for grp in ("relevant", "nonrelevant"):
                rs = [r for r in records if r["qid"] == qid and r["relevance_group"] == grp]
                trimmed += rs if k is None else rs[:int(k)]
        records = trimmed
        summary = [dict(s, n_q_used=sum(1 for r in records if r["qid"] == s["qid"]) // 2) for s in summary
                   if s["qid"] in keep_q]
    else:
        summary = [dict(s, n_q_used=s["n_q"]) for s in summary]

    for r in records:
        r["query"] = queries[r["qid"]]
        r["passage"] = texts[r["docid"]]
        r["pair_id"] = pair_id(r["qid"], r["docid"])
        r["in_bm25_top1000"] = r["docid"] in bm_t
        seq = encode_clean(tok, r["query"], r["passage"], max_len)
        r.update({"seq_len": seq.seq_len, "n_query_tokens": seq.n_query_tokens, "n_document_tokens": seq.n_doc_tokens})

    grades = Counter(g for _, _, g in qrels)
    prov["qrels"] = {
        "ir_datasets_version": ir_datasets.__version__,
        "irds_qrels_id": qc["irds_qrels_id"], "qrels_local_file": str(qrels_path),
        "qrels_local_sha256": file_sha256(qrels_path) if qrels_path.exists() else None,
        "qrels_origin_url": "https://trec.nist.gov/data/deep/2019qrels-pass.txt",
        "upstream_usage": "ecir24-adversarial-evaluation/advseq2seq/retrieval_effectiveness/evaluation_utils.py "
                          "pt.get_dataset('irds:msmarco-passage/trec-dl-2019/judged')",
        "n_qrels": len(qrels), "grade_counts": {str(k): v for k, v in sorted(grades.items())},
        "n_judged_queries": len(judged_qids),
        "groups": {"relevant": [2, 3], "nonrelevant": [0], "excluded": [1]},
        "balancing": "per query n_q=min(|R|,|N|); smaller class kept whole; larger class sampled with "
                     "random.Random(f'42:{qid}') over sorted docids",
        "doc_text_source": f"ir_datasets '{qc['irds_docs_id']}' docs_iter (MS MARCO passage v1 collection)",
        "doc_text_cache": str(text_cache),
        "n_judged_docids": len(needed),
        "crosscheck_bm25_run_text": {"file": cfg["data"]["bm25_run_text"], "n_judged_docids_in_bm25_top1000": len(overlap),
                                     "n_identical_text": same_text},
        "crosscheck_query_text": {"n_judged_queries": len(judged_qids), "n_identical_to_bm25_run": q_same},
        "n_too_long_excluded_before_balancing": len(too_long), "too_long": too_long[:50],
        "n_queries_retained": len({r["qid"] for r in records}),
        "n_selected_relevant": sum(r["relevance_group"] == "relevant" for r in records),
        "n_selected_nonrelevant": sum(r["relevance_group"] == "nonrelevant" for r in records),
        "smoke_trimmed": qc.get("max_queries") is not None or qc.get("max_docs_per_class") is not None,
    }
    return records, summary


def main():
    args = stage_argparser("Exp 16 stage 00: manifests").parse_args()
    cfg = load_config(args.config)
    out = output_dir(cfg)
    mdir = out / "manifests"
    required = ["clean.jsonl", "attacks.jsonl.gz", "qrel_balanced.jsonl", "qrel_query_summary.csv", "provenance.json"]
    if not args.force and is_already_successful(mdir, required):
        print(f"[00] manifests already complete in {mdir} (use --force to rebuild)")
        return
    write_status(mdir, {"status": "running", "started": now()})
    tok = load_tokenizer(cfg)
    prov = {"config": cfg["_config_path"], "attack_grid_source": cfg["_attack_grid_source"], "created": now(), "model": cfg["model"]["checkpoint"]}

    clean, canonical = build_clean(cfg, tok, prov)
    attacks = build_attacks(cfg, tok, canonical, prov)
    qrel, qsummary = build_qrels(cfg, tok, prov, out)
    cap = cfg["data"].get("max_clean_examples")
    if cap is not None:
        clean = clean[:int(cap)]
        prov["clean"]["max_clean_examples"] = int(cap)

    write_jsonl(mdir / "clean.jsonl", clean)
    write_jsonl(mdir / "attacks.jsonl.gz", attacks)
    write_jsonl(mdir / "qrel_balanced.jsonl", qrel)
    pd.DataFrame(qsummary).to_csv(mdir / "qrel_query_summary.csv", index=False)
    pd.DataFrame(prov["attacks"]["per_attack"]).to_csv(mdir / "attack_population_summary.csv", index=False)
    (mdir / "provenance.json").write_text(json.dumps(prov, indent=2))
    sha = {k: file_sha256(mdir / f) for k, f in
           {"clean": "clean.jsonl", "attacks": "attacks.jsonl.gz", "qrel": "qrel_balanced.jsonl"}.items()}
    write_status(mdir, {"status": "success", "finished": now(), "sha256": sha,
                        "n_clean": len(clean), "n_attack_examples": len(attacks), "n_qrel_docs": len(qrel)})
    print(f"[00] clean={len(clean)} attack_examples={len(attacks)} qrel_docs={len(qrel)} "
          f"(queries={prov['qrels']['n_queries_retained']}) -> {mdir}")


if __name__ == "__main__":
    main()
