"""
exp9lib/decoderlens_import.py
================================
Load DecoderLens's candidate-set and ranking code by file path instead of
via `sys.path` + `import src.X`.

Why: Experiment 1's importable package and DecoderLens's importable package
are BOTH named `src` (top-level). If both directories are added to
sys.path, `import src` resolves to whichever gets imported FIRST and is
then cached in sys.modules — later sys.path reordering can't undo that.
`from src.ranking import compute_rank` could silently return Experiment 1's
`src` package (which has no `ranking` module) instead of DecoderLens's.

Fix: load DecoderLens's two needed files directly from their file paths via
importlib, under module names that don't collide with anything
(`_decoderlens_data_loading`, `_decoderlens_ranking`). Both files are
confirmed to have zero internal imports beyond the stdlib, so loading them
standalone (outside their `src` package) is safe.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys
import types


def _load_module_from_path(module_name: str, file_path: pathlib.Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    # Register under its private name BEFORE exec: the dataclasses module
    # resolves string type annotations via sys.modules[cls.__module__], so
    # skipping this step breaks Candidate's dataclass fields at call time.
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_decoderlens_ranking(decoderlens_dir: pathlib.Path):
    """
    Returns (Candidate, load_attack_tsv, build_candidate_sets, compute_rank)
    loaded directly from monot5_decoderlens_rank/src/{data_loading,ranking}.py,
    bypassing the `src` package name collision with Experiment 1.
    """
    data_loading = _load_module_from_path(
        "_decoderlens_data_loading", decoderlens_dir / "src" / "data_loading.py"
    )
    ranking = _load_module_from_path(
        "_decoderlens_ranking", decoderlens_dir / "src" / "ranking.py"
    )
    return (
        data_loading.Candidate,
        data_loading.load_attack_tsv,
        data_loading.build_candidate_sets,
        ranking.compute_rank,
    )
