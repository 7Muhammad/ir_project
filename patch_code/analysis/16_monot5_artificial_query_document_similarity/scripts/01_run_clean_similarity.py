"""
scripts/01_run_clean_similarity.py
===================================
Stage 01 — (a) real-model sanity checks, (b) clean Type-A similarities.

(a) On the first `validation.sanity_n_examples` clean pairs (+ the first
    attack/control pair), fail loudly unless:
      * the hooks capture exactly 25 [B, S, 768] states with the documented
        semantics (exp16lib.sanity.assert_hook_semantics);
      * batching with padding does not change similarities (> batch_invariance_atol);
      * (smoke) freshly computed Exp 01 clean scores match the cached ones.
(b) sim_clean(c) = cos(mean query-text states, mean passage states) for every
    canonical pair and checkpoint; the cached Exp 01 original_score is attached.

Output: 01_clean/clean_similarity.csv, 01_clean/sanity_report.json
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

EXP_DIR = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EXP_DIR))
import exp16lib  # noqa: E402,F401

from exp16lib.checkpoints import CHECKPOINTS  # noqa: E402
from exp16lib.config import load_config, output_dir  # noqa: E402
from exp16lib.engine import run_sequences  # noqa: E402
from exp16lib.inputs import collate, encode_attack_and_control, encode_clean  # noqa: E402
from exp16lib.run_utils import (load_manifest, load_model, now, read_jsonl, stage_argparser,  # noqa: E402
                                unit_is_done, write_status)
from exp16lib.sanity import assert_hook_semantics, batch_invariance_report  # noqa: E402
from exp16lib.tables import long_table  # noqa: E402
from src.model_utils import build_monot5_input  # noqa: E402
from src.scoring import score_batch  # noqa: E402


def main():
    args = stage_argparser("Exp 16 stage 01: sanity + clean similarity").parse_args()
    cfg = load_config(args.config)
    out = output_dir(cfg)
    clean, sha = load_manifest(out, "clean")
    stage = out / "01_clean"
    if unit_is_done(stage, ["clean_similarity.csv", "sanity_report.json"], sha, args.force):
        print(f"[01] already complete ({stage})")
        return
    write_status(stage, {"status": "running", "started": now()})
    val = cfg["validation"]
    max_len = int(cfg["model"]["max_length"])
    model, tok, true_id, false_id, device = load_model(cfg)
    pad = tok.pad_token_id
    seqs = [encode_clean(tok, r["query"], r["passage"], max_len) for r in clean]
    for r, s in zip(clean, seqs):
        if (s.seq_len, s.n_query_tokens, s.n_doc_tokens) != (r["seq_len"], r["n_query_tokens"], r["n_document_tokens"]):
            raise RuntimeError(f"{r['pair_id']}: rebuilt encoding differs from manifest")

    # ---- (a) sanity ------------------------------------------------------
    n_s = min(int(val["sanity_n_examples"]), len(seqs))
    atk0 = read_jsonl(out / "manifests" / "attacks.jsonl.gz")[0]
    a_seq, c_seq, _ = encode_attack_and_control(tok, atk0["query"], atk0["passage"], atk0["attacked_passage"], max_len)
    sanity_seqs = seqs[:n_s] + [a_seq, c_seq]   # includes a padded control (internal mask zeros)
    rep = assert_hook_semantics(model, collate(sanity_seqs, pad, device), float(val["hook_semantics_atol"]),
                                expected_checkpoints=len(CHECKPOINTS))
    rep["batch_invariance_max_abs"] = batch_invariance_report(model, sanity_seqs, pad, device)
    if rep["batch_invariance_max_abs"] > float(val["batch_invariance_atol"]):
        raise AssertionError(f"batch invariance violated: {rep['batch_invariance_max_abs']}")
    n_fresh = min(int(val.get("fresh_score_check_n", 0)), len(clean))
    if n_fresh:
        fresh = score_batch(model, tok, [build_monot5_input(r["query"], r["passage"]) for r in clean[:n_fresh]],
                            true_id, false_id, max_len, device, batch_size=n_fresh)
        diffs = [abs(f - r["score_clean"]) for f, r in zip(fresh, clean[:n_fresh])]
        rep["fresh_vs_cached_clean_score"] = {"n": n_fresh, "max_abs_diff": max(diffs),
                                              "fresh": fresh, "cached": [r["score_clean"] for r in clean[:n_fresh]]}
        if max(diffs) > float(val["fresh_vs_cached_score_atol"]):
            raise AssertionError(f"fresh clean scores disagree with the Exp 01 cache: {diffs}")
    rep["device"] = str(device)
    (stage / "sanity_report.json").write_text(json.dumps(rep, indent=2))
    print("[01] sanity OK:", json.dumps({k: v for k, v in rep.items() if not isinstance(v, dict)}))

    # ---- (b) clean similarities ------------------------------------------
    sims, _ = run_sequences(model, seqs, pad, device, int(cfg["runtime"]["batch_size"]))
    meta = [{k: r[k] for k in ("pair_id", "qid", "docid", "query", "score_clean")} |
            {"n_query_tokens": s.n_query_tokens, "n_document_tokens": s.n_doc_tokens} for r, s in zip(clean, seqs)]
    df = long_table(meta, {"similarity": sims})
    df.to_csv(stage / "clean_similarity.csv", index=False)
    write_status(stage, {"status": "success", "finished": now(), "manifest_sha256": sha, "n_examples": len(clean),
                         "n_rows": len(df), "device": str(device), "sim_min": float(np.min(sims)),
                         "sim_max": float(np.max(sims))})
    print(f"[01] {len(clean)} clean examples x {sims.shape[1]} checkpoints -> {stage / 'clean_similarity.csv'}")


if __name__ == "__main__":
    main()
