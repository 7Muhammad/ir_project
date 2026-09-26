"""
src/find_data.py
================
Data-discovery utilities for the monoT5 activation-patching experiment.

The goal is to locate four file types that already exist somewhere on disk:
  1. prepared pairs    – JSONL/CSV/TSV with (qid, docid, query, passage)
  2. queries file      – DL19 queries (TSV or TXT)
  3. passages/docstore – MS MARCO passage collection (TSV / JSONL / sqlite …)
  4. BM25 run file     – TREC run format  "qid Q0 docid rank score tag"

Priority:
  1. prepared_pairs_path  (use as-is, skip everything else)
  2. upstream_repo_path   (look inside the cloned ECIR-24 repo)
  3. search_roots         (walk the configured directories)

We never recompute BM25 — we only read pre-existing files.
"""

from __future__ import annotations

import os
import pathlib
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# File name patterns that indicate each category
# ---------------------------------------------------------------------------

# Patterns for prepared query-document pairs (already have both query & text)
PAIRS_PATTERNS = [
    "bm25_19.jsonl",   # the file that already exists in the original repo!
    "pairs.jsonl",
    "bm25_19",
    "dl19_pairs",
    "dl19-pairs",
]

# Patterns for raw DL19 queries files
QUERY_PATTERNS = [
    "dl19-queries",
    "dl19_queries",
    "msmarco-test2019-queries",
    "queries.dl19",
    "topics.dl19",
    "2019.queries",
]

# Patterns for MS MARCO passage collection
PASSAGE_PATTERNS = [
    "collection.tsv",
    "passages.tsv",
    "msmarco-passages",
    "collection",
    "docstore",
]

# Patterns for BM25 TREC run files
BM25_RUN_PATTERNS = [
    "dl19-baseline-bm25.trec",   # found in the ECIR-24 repo
    "bm25_19.trec",
    "baseline_bm25",
    "run.bm25",
    "bm25.run",
]

# Patterns for qrels files
QREL_PATTERNS = [
    "qrels.dl19",
    "2019qrels",
    "dl19-qrels",
    "dl19_qrels",
]

# File extensions we look at
VALID_EXTENSIONS = {
    ".jsonl", ".json", ".tsv", ".csv", ".txt", ".res", ".run", ".gz"
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _matches_any(name: str, patterns: List[str]) -> bool:
    """Return True if `name` contains any of the given pattern substrings."""
    name_lower = name.lower()
    return any(p.lower() in name_lower for p in patterns)


def _walk_root(root: pathlib.Path, max_depth: int = 6) -> List[pathlib.Path]:
    """
    Yield all files under `root` up to `max_depth` levels deep.
    Skips hidden directories (those starting with '.') and common
    irrelevant directories like __pycache__, node_modules, etc.
    """
    skip_dirs = {".git", "__pycache__", "node_modules", ".venv", "venv", ".tox"}
    results: List[pathlib.Path] = []

    def _recurse(path: pathlib.Path, depth: int) -> None:
        if depth > max_depth:
            return
        try:
            for child in path.iterdir():
                if child.is_dir():
                    if child.name not in skip_dirs and not child.name.startswith("."):
                        _recurse(child, depth + 1)
                elif child.is_file():
                    # Only consider files with relevant extensions
                    suffix = child.suffix.lower()
                    # .gz files keep their "real" extension inside
                    if suffix in VALID_EXTENSIONS:
                        results.append(child)
        except PermissionError:
            pass  # skip dirs we can't read

    _recurse(root, 0)
    return results


def _classify_files(
    files: List[pathlib.Path],
) -> Dict[str, List[pathlib.Path]]:
    """
    Group discovered files into four categories.

    Returns a dict with keys:
      "pairs", "queries", "passages", "bm25_run", "qrels"
    """
    cats: Dict[str, List[pathlib.Path]] = {
        "pairs": [],
        "queries": [],
        "passages": [],
        "bm25_run": [],
        "qrels": [],
    }
    for f in files:
        name = f.name
        if _matches_any(name, PAIRS_PATTERNS):
            cats["pairs"].append(f)
        if _matches_any(name, QUERY_PATTERNS):
            cats["queries"].append(f)
        if _matches_any(name, PASSAGE_PATTERNS):
            cats["passages"].append(f)
        if _matches_any(name, BM25_RUN_PATTERNS):
            cats["bm25_run"].append(f)
        if _matches_any(name, QREL_PATTERNS):
            cats["qrels"].append(f)
    return cats


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def discover_data(
    upstream_repo_path: Optional[str],
    search_roots: List[str],
    report_path: pathlib.Path,
    script_dir: Optional[pathlib.Path] = None,
) -> Dict[str, List[pathlib.Path]]:
    """
    Walk the configured directories and classify candidate files by type.

    The search order is:
      1. upstream_repo_path  (if set)
      2. each directory in search_roots, resolved relative to script_dir if
         the path is relative

    Results are written to `report_path` (outputs/data_discovery/data_candidates.txt)
    and returned as a dict with lists of Path objects per category.

    Parameters
    ----------
    upstream_repo_path:
        Absolute path to the cloned ECIR-24 repository, or None.
    search_roots:
        List of directory paths to walk.  Relative paths are resolved
        relative to `script_dir`.
    report_path:
        Where to write the human-readable candidate file report.
    script_dir:
        Directory of the calling script, used to resolve relative search roots.
        Defaults to the current working directory.

    Returns
    -------
    Dict mapping category name → list of candidate Path objects.
    """
    if script_dir is None:
        script_dir = pathlib.Path.cwd()

    all_files: List[pathlib.Path] = []

    # --- Search upstream repo first -----------------------------------------
    roots_to_search: List[pathlib.Path] = []
    if upstream_repo_path:
        p = pathlib.Path(upstream_repo_path).expanduser().resolve()
        if p.is_dir():
            roots_to_search.append(p)
            print(f"[find_data] Searching upstream repo: {p}")
        else:
            print(f"[find_data] WARNING: upstream_repo_path '{p}' is not a directory")

    # --- Then search roots --------------------------------------------------
    for root_str in search_roots:
        p = pathlib.Path(root_str).expanduser()
        if not p.is_absolute():
            p = (script_dir / p).resolve()
        else:
            p = p.resolve()
        if p.is_dir():
            roots_to_search.append(p)
            print(f"[find_data] Searching root: {p}")
        else:
            print(f"[find_data] WARNING: search root '{p}' is not a directory, skipping")

    for root in roots_to_search:
        found = _walk_root(root)
        all_files.extend(found)

    # Deduplicate (same resolved path from multiple roots)
    all_files = list({f.resolve(): f for f in all_files}.values())

    # Classify into categories
    candidates = _classify_files(all_files)

    # --- Write report -------------------------------------------------------
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as fh:
        fh.write("# Data Discovery Report\n")
        fh.write(f"# Searched roots: {[str(r) for r in roots_to_search]}\n")
        fh.write(f"# Total files scanned: {len(all_files)}\n\n")
        for cat, paths in candidates.items():
            fh.write(f"## {cat.upper()} ({len(paths)} candidates)\n")
            for p in sorted(paths):
                fh.write(f"  {p}\n")
            fh.write("\n")

    # Print summary to console
    print("\n[find_data] Discovery summary:")
    for cat, paths in candidates.items():
        print(f"  {cat:12s}: {len(paths)} candidate(s)")
    print(f"\n[find_data] Full report written to: {report_path}")

    return candidates


def resolve_single_candidate(
    candidates: List[pathlib.Path],
    category: str,
    config_override: Optional[str] = None,
) -> Optional[pathlib.Path]:
    """
    Return the single best candidate for a given category, or None.

    If `config_override` is set, it takes precedence.

    Rules:
    - If config_override is set → return that path (error if it does not exist).
    - If exactly one candidate   → return it.
    - If zero candidates         → return None (caller decides how to handle).
    - If multiple candidates     → print them all and return None so the caller
      can stop and ask the user to set the override manually.

    This function deliberately does NOT silently pick between ambiguous files.
    """
    if config_override:
        p = pathlib.Path(config_override).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(
                f"Config override for '{category}' does not exist: {p}"
            )
        return p

    if len(candidates) == 1:
        return candidates[0]

    if len(candidates) == 0:
        return None

    # Multiple candidates — do not guess
    print(f"\n[find_data] AMBIGUOUS: found {len(candidates)} candidates for '{category}':")
    for c in sorted(candidates):
        print(f"  {c}")
    print(
        f"\n[find_data] ACTION REQUIRED: set the '{category}' path explicitly in "
        f"configs/default.yaml to resolve the ambiguity.\n"
    )
    return None
