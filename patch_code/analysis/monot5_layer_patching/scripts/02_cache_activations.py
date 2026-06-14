#!/usr/bin/env python3
"""
scripts/02_cache_activations.py
================================
Stage 02: Run control and attack forward passes and cache all layer activations.

Input
-----
  outputs/scores/selected_examples.jsonl   (produced by Stage 01)

Outputs
-------
  outputs/activations/{qid}_{docid}_control.pt
  outputs/activations/{qid}_{docid}_attack.pt

Each .pt file is a dict with two keys:
  "activations" : { "component_layer{i}": Tensor(1, seq_len, d_model), ... }
  "encoding"    : { "input_ids": Tensor(1, L), "attention_mask": Tensor(1, L) }

Why we cache both activations AND encodings
--------------------------------------------
Stage 03 (patching) needs to run forward passes with the pre-built encoding
dicts (not just text strings), because the padded-control encoding carries a
custom attention_mask with 0s for the pad-token slots.  Saving the encoding
alongside the activations means Stage 03 never needs to rebuild them.

Why we cache activations
-------------------------
Activation patching (Stage 03) requires us to:
  1. Do a forward pass with one input and save intermediate activations.
  2. Do another forward pass with a different input, but mid-way through
     replace one layer's activation with the saved one.

Caching separates step 1 (expensive: run all layers for both inputs) from
step 2 (inject saved activation + run final layers).  This avoids redundant
computation when sweeping over many (layer, component) combinations.

Memory note
-----------
Each .pt file for monoT5-base with 512-token sequences contains:
  12 enc_self_attn + 12 enc_mlp + 12 dec_self_attn + 12 dec_cross_attn + 12 dec_mlp
  = 60 tensors × (1 × seq_len × 768) float32
  ≈ 60 × 1.5 MB = ~90 MB per example (on disk, much less with compression).

To limit memory, max_cached_examples is enforced from the config.
All cached tensors are stored on CPU.

Why we use torch.no_grad()
---------------------------
We are doing inference only — no gradients are needed.  torch.no_grad() prevents
PyTorch from building a computation graph, which saves memory and speeds up the
forward passes significantly.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

SCRIPT_DIR = pathlib.Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import yaml

from src.model_utils import (
    build_padded_control_and_attack_encodings,
    get_true_false_token_ids,
    load_monot5,
    resolve_device,
)
from src.activation_hooks import cache_all_activations_from_enc


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stage 02: cache control and attack activations for selected examples."
    )
    p.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "default.yaml"),
        help="Path to the YAML config file.",
    )
    p.add_argument(
        "--attack-name",
        default=None,
        help="If set, read/write outputs under outputs/attacks/{attack-name}/.",
    )
    return p.parse_args()


def _resolve_base_dir(cfg: dict, attack_name: str | None) -> pathlib.Path:
    base = PROJECT_ROOT / cfg["outputs"]["base_dir"]
    if attack_name:
        return base / "attacks" / attack_name
    return base


def main() -> None:
    args = parse_args()
    cfg_path = pathlib.Path(args.config).resolve()
    print(f"[02_cache_activations] Loading config from: {cfg_path}")

    with open(cfg_path) as fh:
        cfg = yaml.safe_load(fh)

    model_cfg   = cfg["model"]
    attack_cfg  = cfg["attack"]
    runtime_cfg = cfg["runtime"]

    base_dir   = _resolve_base_dir(cfg, args.attack_name)
    selected_path = base_dir / "scores" / "selected_examples.jsonl"
    act_dir    = base_dir / "activations"
    act_dir.mkdir(parents=True, exist_ok=True)

    if not selected_path.exists():
        sys.exit(
            f"ERROR: selected_examples.jsonl not found at {selected_path}.\n"
            "Please run Stage 01 first:  python scripts/01_score_and_select.py"
        )

    max_cached     = runtime_cfg["max_cached_examples"]
    max_length     = model_cfg["max_length"]

    # Load selected examples
    examples = []
    with open(selected_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    print(f"[02_cache_activations] Loaded {len(examples)} selected examples.")

    # Enforce max_cached_examples limit
    if len(examples) > max_cached:
        examples = examples[:max_cached]
        print(f"[02_cache_activations] Truncated to {max_cached} (max_cached_examples).")

    # Load model
    device = resolve_device(model_cfg["device"])
    model, tokenizer = load_monot5(model_cfg["checkpoint"], device)

    # Cache activations for each selected example
    for i, ex in enumerate(examples):
        qid    = ex["qid"]
        docid  = ex["docid"]
        query  = ex["query"]
        passage = ex["passage"]

        print(
            f"[02_cache_activations] [{i+1}/{len(examples)}] "
            f"qid={qid} docid={docid}"
        )

        # Build padded-control and attacked encodings simultaneously.
        # Both have the same sequence length; the padded-control has
        # attention_mask=0 for the attack-prefix positions.
        control_enc, attack_enc, n_prefix, n_attack_tokens = (
            build_padded_control_and_attack_encodings(
                tokenizer=tokenizer,
                query=query,
                passage=passage,
                attacked_passage=ex["attacked_passage"],
                max_length=max_length,
                device=device,
            )
        )
        print(
            f"  seq_len={attack_enc['input_ids'].shape[1]}  "
            f"common_prefix={n_prefix}  attack_tokens={n_attack_tokens}"
        )

        # Cache control activations from the padded-control encoding
        control_cache = cache_all_activations_from_enc(
            model=model,
            enc=control_enc,
            device=device,
        )
        control_out = act_dir / f"{qid}_{docid}_control.pt"
        torch.save(
            {"activations": control_cache,
             "encoding": {k: v.cpu() for k, v in control_enc.items()}},
            control_out,
        )

        # Cache attack activations from the attacked encoding
        attack_cache = cache_all_activations_from_enc(
            model=model,
            enc=attack_enc,
            device=device,
        )
        attack_out = act_dir / f"{qid}_{docid}_attack.pt"
        torch.save(
            {"activations": attack_cache,
             "encoding": {k: v.cpu() for k, v in attack_enc.items()}},
            attack_out,
        )

        print(
            f"  → saved control: {control_out.name}  "
            f"attack: {attack_out.name}  "
            f"({len(control_cache)} tensors each)"
        )

    print(f"\n[02_cache_activations] Done. Activations saved to: {act_dir}")


if __name__ == "__main__":
    main()
