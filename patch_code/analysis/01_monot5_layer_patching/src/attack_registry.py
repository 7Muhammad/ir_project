"""
src/attack_registry.py
======================
Discovery and parsing of ECIR-24 manual injection attack TSV files.

File naming convention (upstream repo)
---------------------------------------
    {token}_{position}_{repetitions}_{run}.gz.tsv

Examples:
    relevant_start_5_bm25_19.gz.tsv
    informationtrue_end_3_bm25_19.gz.tsv
    relevantfalse_random_2_bm25_19.gz.tsv

IMPORTANT: ``token`` may contain underscores (e.g. "relevantbar", but also
potentially multi-word tokens).  We parse right-to-left anchored on the
closed vocabularies for ``position`` and ``repetitions`` so the token can
be anything:

    regex: ^(?P<token>.+)_(?P<position>start|end|random)_(?P<reps>[1-9]\\d*)_(?P<run>.+)\\.gz\\.tsv$

Config modes
------------
  mode: "include"   — only the named attack_names in ``include`` list
  mode: "pattern"   — glob pattern matched against attack_name
  mode: "all"       — every file found in upstream_injected_dir

Usage
-----
    from src.attack_registry import discover_attacks, AttackSpec

    attacks = discover_attacks(cfg["attacks"])
    for spec in attacks:
        print(spec.attack_name, spec.path)
"""

from __future__ import annotations

import fnmatch
import pathlib
import re
from dataclasses import dataclass
from typing import List, Optional


# ---------------------------------------------------------------------------
# Regex — anchored on the closed-vocabulary fields (position, reps)
# so the ``token`` field absorbs everything to the left.
# ---------------------------------------------------------------------------

_PATTERN = re.compile(
    r"^(?P<token>.+)"
    r"_(?P<position>start|end|random)"
    r"_(?P<reps>[1-9]\d*)"
    r"_(?P<run>.+)"
    r"\.gz\.tsv$"
)


@dataclass
class AttackSpec:
    """Parsed metadata for one injection attack file."""

    attack_name: str          # e.g. "relevant_start_5"
    token: str                # e.g. "relevant"
    position: str             # "start" | "end" | "random"
    repetitions: int          # 1-5 (or whatever the file has)
    run_name: str             # e.g. "bm25_19"
    path: pathlib.Path        # absolute path to the .gz.tsv file


def parse_attack_filename(filename: str) -> Optional[AttackSpec]:
    """
    Parse one attack filename into an AttackSpec.

    Parameters
    ----------
    filename : str
        Basename of the file (e.g. "relevant_start_5_bm25_19.gz.tsv").

    Returns
    -------
    AttackSpec if the filename matches the expected pattern, else None.
    """
    m = _PATTERN.match(filename)
    if m is None:
        return None
    token      = m.group("token")
    position   = m.group("position")
    reps       = int(m.group("reps"))
    run_name   = m.group("run")
    attack_name = f"{token}_{position}_{reps}"
    return AttackSpec(
        attack_name=attack_name,
        token=token,
        position=position,
        repetitions=reps,
        run_name=run_name,
        path=pathlib.Path(),  # filled in by discover_attacks
    )


def _all_specs_in_dir(directory: pathlib.Path) -> List[AttackSpec]:
    """Return a parsed AttackSpec for every matching .gz.tsv file in directory."""
    specs: List[AttackSpec] = []
    if not directory.is_dir():
        return specs
    for fpath in sorted(directory.iterdir()):
        if not fpath.is_file():
            continue
        spec = parse_attack_filename(fpath.name)
        if spec is not None:
            spec.path = fpath.resolve()
            specs.append(spec)
    return specs


def discover_attacks(attacks_cfg: dict) -> List[AttackSpec]:
    """
    Discover attack TSV files according to the ``attacks`` config block.

    Config block structure
    ----------------------
    attacks:
      upstream_injected_dir: "/path/to/runs/injected/dl19"
      mode: "include"         # "include" | "pattern" | "all"
      include:                # used when mode == "include"
        - relevant_start_5
        - true_start_5
      include_pattern: null   # used when mode == "pattern"
      max_attacks: null       # hard cap on number returned

    Parameters
    ----------
    attacks_cfg : dict
        The ``attacks:`` block from the YAML config.

    Returns
    -------
    List[AttackSpec] in the order determined by the mode.

    Raises
    ------
    FileNotFoundError
        If upstream_injected_dir does not exist.
    ValueError
        If mode is invalid or a named attack is not found (mode == "include").
    """
    injected_dir_str = attacks_cfg.get("upstream_injected_dir", "")
    if not injected_dir_str:
        raise ValueError(
            "attacks.upstream_injected_dir must be set in the config.\n"
            "Example: /home/.../ecir24-adversarial-evaluation/runs/injected/dl19"
        )
    injected_dir = pathlib.Path(injected_dir_str).resolve()
    if not injected_dir.exists():
        raise FileNotFoundError(
            f"attacks.upstream_injected_dir does not exist: {injected_dir}"
        )

    all_specs = _all_specs_in_dir(injected_dir)
    # Build lookup by attack_name (e.g. "relevant_start_5")
    by_name = {s.attack_name: s for s in all_specs}

    mode = attacks_cfg.get("mode", "include")
    max_attacks: Optional[int] = attacks_cfg.get("max_attacks")

    if mode == "all":
        result = all_specs

    elif mode == "include":
        include_list = attacks_cfg.get("include") or []
        if not include_list:
            raise ValueError(
                "attacks.mode is 'include' but attacks.include is empty. "
                "Add attack names or switch to mode: 'all'."
            )
        result = []
        missing = []
        for name in include_list:
            if name in by_name:
                result.append(by_name[name])
            else:
                missing.append(name)
        if missing:
            raise ValueError(
                f"The following attack names were listed in attacks.include "
                f"but no matching file was found in {injected_dir}:\n"
                + "\n".join(f"  - {n}" for n in missing)
                + "\nAvailable attack_names (first 20):\n"
                + "\n".join(f"  - {s.attack_name}" for s in all_specs[:20])
            )

    elif mode == "pattern":
        pattern = attacks_cfg.get("include_pattern")
        if not pattern:
            raise ValueError(
                "attacks.mode is 'pattern' but attacks.include_pattern is null/empty."
            )
        result = [s for s in all_specs if fnmatch.fnmatch(s.attack_name, pattern)]
        if not result:
            raise ValueError(
                f"Pattern '{pattern}' matched no attack_names in {injected_dir}."
            )

    else:
        raise ValueError(
            f"Unknown attacks.mode: '{mode}'. Expected 'include', 'pattern', or 'all'."
        )

    if max_attacks is not None:
        result = result[:max_attacks]

    return result
