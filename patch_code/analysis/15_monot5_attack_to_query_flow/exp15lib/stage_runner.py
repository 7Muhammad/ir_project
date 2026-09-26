"""
exp15lib/stage_runner.py
=========================
Shared per-attack loop for the model stages (02 all-query edges, 03
single-query-token edges, 04 whole-head reference).

Resume unit = one attack (all selected examples x all heads x both
directions). Each unit writes `{stage_dir}/per_attack/{attack}/rows.csv.gz`
+ status.json recording the manifest hash and head labels; a unit is skipped
only if both match (exp15lib.run_utils.unit_is_done). Chunking across SLURM
jobs: `--attack-start/--attack-end` select a deterministic slice of the
manifest's attack order, so several jobs can split the grid without
overlapping; stage 05 refuses to aggregate until every unit is complete.

For every example the encodings are REBUILT from the manifest text with the
same Exp 01/06 machinery and must reproduce the manifest's A, Q and query
token ids exactly (fail loudly otherwise).
"""

from __future__ import annotations

import argparse
import pathlib
import time
from typing import Callable, Dict, List

import pandas as pd
import torch

import exp15lib  # noqa: F401
from exp15lib.config import load_config, output_dir
from exp15lib.heads import load_heads
from exp15lib.positions import build_encodings_and_positions
from exp15lib.run_utils import load_manifest, load_model, now, unit_is_done, write_status

META_COLS = [
    "attack_name", "attack_token", "attack_position", "repetitions", "sample_index",
    "example_id", "qid", "docid", "alignment_boundary_shift",
    "n_attack_positions", "n_query_tokens", "seq_len",
    "score_control", "score_attack", "delta",
]


def stage_argparser(description: str, exp_dir: pathlib.Path) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", default=str(exp_dir / "configs" / "default.yaml"))
    p.add_argument("--force", action="store_true")
    p.add_argument("--attack-start", type=int, default=0, help="first attack index (grid order) to process")
    p.add_argument("--attack-end", type=int, default=None, help="one past the last attack index")
    return p


def rebuild_encodings(tokenizer, rec: Dict, max_len: int, device: torch.device):
    c, a, info = build_encodings_and_positions(
        tokenizer, rec["query"], rec["passage"], rec["attacked_passage"], max_len, device)
    for k in ("attack_source_positions", "query_target_positions", "query_token_ids", "seq_len"):
        if info[k] != rec[k]:
            raise RuntimeError(f"{rec['attack_name']} {rec['example_id']}: rebuilt {k} != manifest")
    return c, a


def run_stage(
    args: argparse.Namespace,
    stage_name: str,
    per_example: Callable,   # (ctx: dict, rec: dict, control_enc, attack_enc) -> List[dict]
) -> None:
    cfg = load_config(args.config)
    out = output_dir(cfg)
    manifest, sha = load_manifest(out)
    heads = load_heads(cfg)
    labels = [h.label for h in heads]
    stage_dir = out / stage_name

    attack_order: List[str] = []
    for r in manifest:
        if r["attack_name"] not in attack_order:
            attack_order.append(r["attack_name"])
    selected = attack_order[args.attack_start:args.attack_end]
    pending = [a for a in selected
               if not unit_is_done(stage_dir / "per_attack" / a, ["rows.csv.gz"], sha, labels, args.force)]
    print(f"[{stage_name}] {len(selected)} attacks selected, {len(selected) - len(pending)} already done, "
          f"{len(pending)} to run; heads={labels}")
    if not pending:
        return

    model, tok, true_id, false_id, device = load_model(cfg)
    ctx = {"cfg": cfg, "model": model, "tok": tok, "true_id": true_id, "false_id": false_id,
           "device": device, "heads": heads}
    max_len = int(cfg["model"]["max_length"])
    t_stage = time.time()
    for k, attack in enumerate(pending):
        unit = stage_dir / "per_attack" / attack
        write_status(unit, {"status": "running", "started": now()})
        t0 = time.time()
        rows: List[Dict] = []
        recs = [r for r in manifest if r["attack_name"] == attack]
        for rec in recs:
            c, a = rebuild_encodings(tok, rec, max_len, device)
            meta = {k2: rec[k2] for k2 in META_COLS}
            for row in per_example(ctx, rec, c, a):
                rows.append({**meta, **row})
        pd.DataFrame(rows).to_csv(unit / "rows.csv.gz", index=False)
        dt = time.time() - t0
        write_status(unit, {
            "status": "success", "finished": now(), "seconds": dt, "n_examples": len(recs),
            "n_rows": len(rows), "manifest_sha256": sha, "head_labels": labels, "device": str(device),
        })
        el = time.time() - t_stage
        print(f"[{stage_name}] {attack:24s} {len(recs)} ex, {len(rows)} rows, {dt:.1f}s "
              f"({k + 1}/{len(pending)}, elapsed {el / 60:.1f} min, "
              f"ETA {el / (k + 1) * (len(pending) - k - 1) / 60:.1f} min)", flush=True)
