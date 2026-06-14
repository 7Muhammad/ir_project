"""
src/data_utils.py
=================
Utilities for loading and normalising the query-document pair data.

The experiment uses BM25 candidates that were produced by the original
ECIR-24 adversarial-evaluation pipeline.  We never recompute BM25 here;
we only read pre-existing files.

Supported input formats
-----------------------
Mode A (preferred): prepared pairs file
    JSONL with one dict per line:
        {"qid": "...", "docid": "...", "query": "...", "passage": "..."}
    or equivalently in CSV/TSV with those column names.
    The "passage" column may also appear as "text" (as in bm25_19.jsonl).

Mode B: TREC run file + separate queries + passage collection
    Standard TREC run format:  qid Q0 docid rank score tag
    Passage collection: TSV "docid\tpassage_text" or JSONL.
"""

from __future__ import annotations

import csv
import gzip
import json
import pathlib
import random
from typing import Dict, Iterator, List, Optional


# ---------------------------------------------------------------------------
# Normalised record type
# ---------------------------------------------------------------------------

PairRecord = Dict  # keys: qid, docid, rank, query, passage (+ optional bm25_score)


# ---------------------------------------------------------------------------
# Mode A: load prepared pairs
# ---------------------------------------------------------------------------

def load_prepared_pairs(
    path: pathlib.Path,
    max_pairs: int,
    max_docs_per_query: int,
    seed: int = 42,
) -> List[PairRecord]:
    """
    Load a prepared pairs file (JSONL, CSV, or TSV) and normalise it.

    The file must contain columns/keys:
        qid, docid, query, passage  (or "text" instead of "passage")

    Sampling:
        If max_docs_per_query < len(docs for a query), we keep the top-ranked
        docs (or sample if rank is not available).
        If total pairs > max_pairs, we truncate after sampling.

    Parameters
    ----------
    path : pathlib.Path
    max_pairs : int
    max_docs_per_query : int
    seed : int

    Returns
    -------
    List of normalised PairRecord dicts.
    """
    random.seed(seed)
    suffix = path.suffix.lower()

    # Handle .gz transparently
    opener = gzip.open if suffix == ".gz" else open
    inner_suffix = pathlib.Path(path.stem).suffix.lower() if suffix == ".gz" else suffix

    records: List[PairRecord] = []

    if inner_suffix in (".jsonl", ".json", ""):
        # Try JSONL first (bm25_19.jsonl has one JSON dict per line)
        with opener(path, "rt", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                records.append(_normalise_record(obj))
    elif inner_suffix in (".csv", ".tsv"):
        delimiter = "\t" if inner_suffix == ".tsv" else ","
        with opener(path, "rt", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh, delimiter=delimiter)
            for row in reader:
                records.append(_normalise_record(dict(row)))
    else:
        raise ValueError(
            f"Unsupported prepared-pairs file format: {path}\n"
            f"Expected .jsonl, .json, .csv, or .tsv (optionally .gz)."
        )

    print(f"[data_utils] Loaded {len(records)} raw records from {path}")
    records = _truncate_per_query(records, max_docs_per_query)
    records = _truncate_total(records, max_pairs, seed)
    print(f"[data_utils] After truncation: {len(records)} pairs")
    return records


def _normalise_record(obj: dict) -> PairRecord:
    """
    Normalise a raw dict into a PairRecord.

    Handles:
    - "text" as an alias for "passage" (as in bm25_19.jsonl)
    - integer qids / docids (convert to str)
    - missing "rank" (set to -1)
    - missing "bm25_score" (set to None)
    """
    passage = obj.get("passage") or obj.get("text") or ""
    return {
        "qid":        str(obj.get("qid", "")),
        "docid":      str(obj.get("docid", "")),
        "rank":       int(obj.get("rank", -1)),
        "bm25_score": obj.get("score") or obj.get("bm25_score"),
        "query":      str(obj.get("query", "")),
        "passage":    passage,
    }


def _truncate_per_query(
    records: List[PairRecord], max_docs_per_query: int
) -> List[PairRecord]:
    """Keep at most max_docs_per_query records per qid (by rank, ascending)."""
    from collections import defaultdict
    by_qid: Dict[str, List[PairRecord]] = defaultdict(list)
    for r in records:
        by_qid[r["qid"]].append(r)
    result = []
    for qid, recs in by_qid.items():
        # Sort by rank; -1 means rank unknown → keep all in original order
        recs_sorted = sorted(recs, key=lambda x: x["rank"] if x["rank"] >= 0 else 999999)
        result.extend(recs_sorted[:max_docs_per_query])
    return result


def _truncate_total(
    records: List[PairRecord], max_pairs: int, seed: int
) -> List[PairRecord]:
    """If more records than max_pairs, sample randomly with fixed seed."""
    if len(records) <= max_pairs:
        return records
    random.seed(seed)
    return random.sample(records, max_pairs)


# ---------------------------------------------------------------------------
# Mode B: TREC run file parser
# ---------------------------------------------------------------------------

def parse_trec_run(
    run_path: pathlib.Path,
    max_docs_per_query: int = 1000,
) -> Dict[str, List[Dict]]:
    """
    Parse a TREC run file into a dict of {qid: [{"docid", "rank", "score"}]}.

    TREC run format (6 space-separated columns):
        qid  Q0  docid  rank  score  tag

    The Q0 and tag fields are ignored.

    Parameters
    ----------
    run_path : pathlib.Path
        Path to the TREC run file (.trec or .trec.gz).
    max_docs_per_query : int
        Keep only the top-N ranked docs per query.

    Returns
    -------
    Dict mapping qid (str) → list of dicts sorted by ascending rank.
    """
    from collections import defaultdict
    results: Dict[str, List] = defaultdict(list)

    opener = gzip.open if run_path.suffix == ".gz" else open
    with opener(run_path, "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 6:
                continue  # malformed line, skip
            qid, _, docid, rank, score, _tag = parts[0], parts[1], parts[2], int(parts[3]), float(parts[4]), parts[5]
            results[qid].append({"docid": docid, "rank": rank, "bm25_score": score})

    # Sort by rank and truncate
    for qid in results:
        results[qid].sort(key=lambda x: x["rank"])
        results[qid] = results[qid][:max_docs_per_query]

    return dict(results)


def load_queries(queries_path: pathlib.Path) -> Dict[str, str]:
    """
    Load DL19 queries from a TSV or TXT file.

    Expected formats:
      TSV: qid<TAB>query_text
      TXT: one query per line (no qid; not supported for reconstruction)

    Returns
    -------
    Dict mapping qid (str) → query text (str).
    """
    queries: Dict[str, str] = {}
    opener = gzip.open if queries_path.suffix == ".gz" else open
    with opener(queries_path, "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) == 2:
                queries[parts[0]] = parts[1]
            else:
                # Single-column: can't associate to qids; warn
                print(f"[data_utils] WARNING: queries file line has no tab: {line[:80]}")
    print(f"[data_utils] Loaded {len(queries)} queries from {queries_path}")
    return queries


def load_passage_collection(
    passages_path: pathlib.Path,
    needed_docids: Optional[set] = None,
) -> Dict[str, str]:
    """
    Load the MS MARCO passage collection (docid → text).

    Supported formats:
      TSV: docid<TAB>passage_text
      JSONL: {"id": "...", "contents": "..."}  or {"docid": "...", "text": "..."}

    If `needed_docids` is provided, only load those documents
    (significant speedup for large collections).

    Parameters
    ----------
    passages_path : pathlib.Path
    needed_docids : set of str, optional

    Returns
    -------
    Dict mapping docid (str) → passage text (str).
    """
    passages: Dict[str, str] = {}
    opener = gzip.open if passages_path.suffix == ".gz" else open
    suffix = pathlib.Path(passages_path.stem).suffix.lower() if passages_path.suffix == ".gz" else passages_path.suffix.lower()

    print(f"[data_utils] Loading passage collection from {passages_path} ...")
    with opener(passages_path, "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if suffix in (".jsonl", ".json", ""):
                obj = json.loads(line)
                docid = str(obj.get("id") or obj.get("docid") or obj.get("pid", ""))
                text = obj.get("contents") or obj.get("text") or obj.get("passage", "")
            else:  # TSV
                parts = line.split("\t", 1)
                if len(parts) < 2:
                    continue
                docid, text = parts[0], parts[1]

            if needed_docids is None or docid in needed_docids:
                passages[docid] = text

    print(f"[data_utils] Loaded {len(passages)} passage texts.")
    return passages


def reconstruct_pairs_from_run(
    run: Dict[str, List[Dict]],
    queries: Dict[str, str],
    passages: Dict[str, str],
    max_pairs: int,
    seed: int = 42,
) -> List[PairRecord]:
    """
    Combine BM25 run + queries + passages into a list of PairRecords.

    Any (qid, docid) combination where either the query text or passage text
    is missing will be silently skipped.

    Parameters
    ----------
    run : output of parse_trec_run
    queries : output of load_queries
    passages : output of load_passage_collection
    max_pairs : int
    seed : int

    Returns
    -------
    List of PairRecord dicts.
    """
    records: List[PairRecord] = []
    for qid, docs in run.items():
        query_text = queries.get(qid)
        if not query_text:
            continue  # query text unavailable, skip
        for doc in docs:
            docid = doc["docid"]
            passage_text = passages.get(docid)
            if not passage_text:
                continue  # passage text unavailable, skip
            records.append({
                "qid":        qid,
                "docid":      docid,
                "rank":       doc["rank"],
                "bm25_score": doc["bm25_score"],
                "query":      query_text,
                "passage":    passage_text,
            })

    records = _truncate_total(records, max_pairs, seed)
    print(f"[data_utils] Reconstructed {len(records)} pairs from run + passages.")
    return records


def save_pairs(pairs: List[PairRecord], output_path: pathlib.Path) -> None:
    """
    Write pairs to a JSONL file (one record per line).

    Each record has the canonical fields:
        qid, docid, rank, bm25_score, query, passage
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        for record in pairs:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"[data_utils] Saved {len(pairs)} pairs to {output_path}")


def load_pairs(pairs_path: pathlib.Path) -> List[PairRecord]:
    """Load a normalised pairs.jsonl file produced by Stage 00."""
    pairs = []
    with open(pairs_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))
    print(f"[data_utils] Loaded {len(pairs)} pairs from {pairs_path}")
    return pairs


def load_attacked_tsv(
    path: pathlib.Path,
    max_pairs: int,
    max_docs_per_query: int,
    seed: int = 42,
) -> List[PairRecord]:
    """
    Load a pre-built attacked document TSV from the ECIR-24 injection pipeline.

    File format (tab-separated, with header row):
        qid  query  docno  score  rank  text  text_0

    Fields:
        text   — attacked passage (e.g. "relevant: "×5 prepended)
        text_0 — original clean passage

    WHY USE PRE-BUILT ATTACKED FILES
    ----------------------------------
    The ECIR-24 repo stores the exact attacked passages that were used in the
    original paper's experiments.  Loading them directly ensures our
    activation patching experiment operates on exactly the same attacked text,
    rather than re-constructing attacks that might differ in tokenization due
    to SentencePiece boundary effects.

    Output record fields:
        qid, docid, rank, bm25_score, query,
        passage          — original clean passage (text_0)
        attacked_passage — attacked passage (text)

    Parameters
    ----------
    path : pathlib.Path
        Path to the attacked TSV file.
    max_pairs : int
        Maximum total pairs to keep.
    max_docs_per_query : int
        Maximum candidates per query (by ascending rank).
    seed : int

    Returns
    -------
    List of PairRecord dicts, each with both 'passage' and 'attacked_passage'.
    """
    records: List[PairRecord] = []

    with open(path, "r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            # The attacked TSV uses "docno" instead of "docid"
            records.append({
                "qid":              str(row.get("qid", "")),
                "docid":            str(row.get("docno") or row.get("docid", "")),
                "rank":             int(row.get("rank", -1)),
                "bm25_score":       row.get("score"),
                "query":            str(row.get("query", "")),
                "passage":          str(row.get("text_0", "")),   # original clean text
                "attacked_passage": str(row.get("text", "")),     # attacked text
            })

    print(f"[data_utils] Loaded {len(records)} rows from attacked TSV: {path.name}")

    records = _truncate_per_query(records, max_docs_per_query)
    records = _truncate_total(records, max_pairs, seed)
    print(f"[data_utils] After truncation: {len(records)} pairs")
    return records
