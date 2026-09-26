"""
exp15lib/run_utils.py
======================
Resume/status helpers (Exp 14's `write_status` / `is_already_successful`,
themselves the Exp 01 status.json pattern), model loading, manifest I/O, and
the manifest-hash guard that keeps every downstream stage tied to ONE
immutable sample.

Resume rule for per-attack units: skip iff status.json says success, the
required files exist, AND the recorded manifest hash + head labels match the
current ones. A mismatch without --force is an error (never silently mixed).
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import time
from typing import Dict, List

import torch

import exp15lib  # noqa: F401
from exp14lib.run_utils import is_already_successful, load_status, write_status  # noqa: F401
from src.model_utils import get_true_false_token_ids, load_monot5, resolve_device

MANIFEST_NAME = "sample_manifest.jsonl"


def load_model(cfg: dict):
    device = usable_device(cfg)
    allow = bool(cfg["runtime"].get("allow_tf32", False))
    torch.backends.cuda.matmul.allow_tf32 = allow
    torch.backends.cudnn.allow_tf32 = allow
    model, tokenizer = load_monot5(cfg["model"]["checkpoint"], device)
    true_id, false_id = get_true_false_token_ids(tokenizer)
    return model, tokenizer, true_id, false_id, device


def usable_device(cfg: dict) -> torch.device:
    """
    `auto` resolution that falls back to CPU if CUDA exists but cannot run
    kernels (the interactive node's Blackwell GPU is unsupported by torch
    2.6/cu124). An explicit `cuda` setting is never silently downgraded.
    """
    dev = resolve_device(cfg["model"]["device"])
    if dev.type == "cuda" and cfg["model"]["device"] == "auto":
        try:
            torch.zeros(1, device=dev) + 1
        except Exception:  # noqa: BLE001
            print("[run_utils] CUDA present but unusable here -> falling back to CPU")
            return torch.device("cpu")
    return dev


def file_sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path: pathlib.Path) -> List[Dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def load_manifest(out_dir: pathlib.Path):
    """Returns (records, sha256). Fails loudly if stage 00 has not completed."""
    stage_dir = out_dir / "00_samples"
    path = out_dir / MANIFEST_NAME
    if not is_already_successful(stage_dir, []) or not path.exists():
        raise FileNotFoundError(f"Sample manifest missing/incomplete ({path}); run scripts/00_prepare_samples.py")
    sha = file_sha256(path)
    recorded = load_status(stage_dir).get("manifest_sha256")
    if recorded != sha:
        raise RuntimeError(f"Manifest {path} changed after stage 00 recorded it (sha mismatch)")
    return read_jsonl(path), sha


def unit_is_done(unit_dir: pathlib.Path, required: List[str], manifest_sha: str, head_labels: List[str],
                 force: bool) -> bool:
    if force or not (unit_dir / "status.json").exists():
        return False
    st = load_status(unit_dir)
    if st.get("status") != "success":
        return False
    if st.get("manifest_sha256") != manifest_sha or st.get("head_labels") != head_labels:
        raise RuntimeError(
            f"{unit_dir} was completed for a different manifest/head set; rerun with --force "
            "(refusing to mix samples)."
        )
    return is_already_successful(unit_dir, required)


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def upstream_fingerprint(files: List[pathlib.Path]) -> str:
    """sha256 over the contents of upstream status/summary files (resume key for cheap stages)."""
    h = hashlib.sha256()
    for f in sorted(files):
        h.update(str(f).encode())
        h.update(pathlib.Path(f).read_bytes())
    return h.hexdigest()


def cheap_stage_is_current(stage_dir: pathlib.Path, fingerprint: str, required: List[str], force: bool) -> bool:
    if force or not is_already_successful(stage_dir, required):
        return False
    return load_status(stage_dir).get("upstream_fingerprint") == fingerprint
