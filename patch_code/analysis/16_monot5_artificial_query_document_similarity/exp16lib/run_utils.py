"""
exp16lib/run_utils.py
======================
Resume/status helpers (Exp 14's `write_status` / `is_already_successful`,
themselves the Exp 01 status.json pattern), model loading, JSONL I/O and a
manifest-hash guard: every model stage records the sha256 of the manifest it
consumed, and a completed unit is reused only if the hash still matches
(otherwise fail loudly unless --force).
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import pathlib
import time
from typing import Dict, List

import torch

import exp16lib  # noqa: F401
from exp14lib.run_utils import is_already_successful, load_status, write_status  # noqa: F401
from src.model_utils import get_true_false_token_ids, load_monot5, resolve_device

EXP_DIR = exp16lib.EXP_DIR


def stage_argparser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", default=str(EXP_DIR / "configs" / "default.yaml"))
    p.add_argument("--force", action="store_true")
    return p


def usable_device(cfg: dict) -> torch.device:
    """`auto` falls back to CPU when CUDA exists but cannot run kernels (Blackwell login node)."""
    dev = resolve_device(cfg["model"]["device"])
    if dev.type == "cuda" and cfg["model"]["device"] == "auto":
        try:
            torch.zeros(1, device=dev) + 1
        except Exception:  # noqa: BLE001
            print("[run_utils] CUDA present but unusable here -> falling back to CPU")
            return torch.device("cpu")
    return dev


def load_model(cfg: dict):
    device = usable_device(cfg)
    allow = bool(cfg["model"].get("allow_tf32", False))
    torch.backends.cuda.matmul.allow_tf32 = allow
    torch.backends.cudnn.allow_tf32 = allow
    model, tokenizer = load_monot5(cfg["model"]["checkpoint"], device)
    true_id, false_id = get_true_false_token_ids(tokenizer)
    return model, tokenizer, true_id, false_id, device


def load_tokenizer(cfg: dict):
    from transformers import T5Tokenizer
    return T5Tokenizer.from_pretrained(cfg["model"]["checkpoint"], use_fast=False)


def file_sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _open(path: pathlib.Path, mode: str):
    return gzip.open(path, mode + "t", encoding="utf-8") if str(path).endswith(".gz") else open(path, mode, encoding="utf-8")


def read_jsonl(path: pathlib.Path) -> List[Dict]:
    with _open(path, "r") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def write_jsonl(path: pathlib.Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with (gzip.open(tmp, "wt", encoding="utf-8") if str(path).endswith(".gz") else open(tmp, "w", encoding="utf-8")) as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    tmp.replace(path)


MANIFEST_FILES = {
    "clean": "clean.jsonl",
    "qrel": "qrel_balanced.jsonl",
    "attacks": "attacks.jsonl.gz",
}


def load_manifest(out_dir: pathlib.Path, kind: str):
    """Returns (records, sha256). Fails loudly if stage 00 is incomplete or the file changed since."""
    mdir = out_dir / "manifests"
    path = mdir / MANIFEST_FILES[kind]
    if not is_already_successful(mdir, [MANIFEST_FILES[kind]]):
        raise FileNotFoundError(f"manifest {path} missing/incomplete; run scripts/00_prepare_manifests.py")
    sha = file_sha256(path)
    if load_status(mdir).get("sha256", {}).get(kind) != sha:
        raise RuntimeError(f"manifest {path} changed after stage 00 recorded it (sha mismatch)")
    return read_jsonl(path), sha


def unit_is_done(unit_dir: pathlib.Path, required: List[str], manifest_sha: str, force: bool) -> bool:
    if force or not (unit_dir / "status.json").exists():
        return False
    st = load_status(unit_dir)
    if st.get("status") != "success":
        return False
    if st.get("manifest_sha256") != manifest_sha:
        raise RuntimeError(f"{unit_dir} was completed for a different manifest; rerun with --force "
                           "(refusing to mix populations).")
    return is_already_successful(unit_dir, required)


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")
