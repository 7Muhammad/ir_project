"""
exp2lib/direction_fit.py
=========================
Mean-diff direction computation for Part 1.

direction = mean(attack_activations) - mean(control_activations)

computed per (component, layer[, head_idx]) at two granularities
(whole_vector, per_head), pooled over the example set for one tier
(grid_a: 105 attacks x 10 examples; grid_b: relevant_start_5 x 100 examples).

Pooling rule (see DECISIONS.md — not specified by the original task):
encoder_self_attn activations carry a real sequence dimension; both
whole-vector and per-head encoder activations are mean-pooled over valid
(attention_mask == 1) positions before being folded into the running mean,
giving one vector per example per (layer[, head]) — the same shape decoder
head activations already have naturally (query length 1, no pooling needed).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch

from headlib.head_hooks import head_geometry

# (component, layer_idx[, head_idx]) — head_idx is None for whole-vector keys.
DirKey = Tuple[str, str, int, Optional[int]]  # (granularity, component, layer, head_idx)


def pool_activation(
    component: str,
    tensor: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Collapse a (1, seq_len, d) activation to a single (d,) vector.

    encoder_self_attn: mean over positions where attention_mask == 1.
    decoder_self_attn / decoder_cross_attn: seq_len is always 1 (single
    decode step) — just squeeze, no pooling needed.
    """
    if component == "encoder_self_attn":
        mask = attention_mask.to(device=tensor.device, dtype=tensor.dtype).unsqueeze(-1)  # (1, seq_len, 1)
        summed = (tensor * mask).sum(dim=1)  # (1, d)
        denom = mask.sum(dim=1).clamp(min=1.0)  # (1, 1)
        return (summed / denom).squeeze(0)  # (d,)
    return tensor.squeeze(0).squeeze(0)  # (1,1,d) -> (d,)


@dataclass
class DirectionAccumulator:
    """Running sums of pooled control/attack activations, keyed by DirKey."""

    n_heads: int
    d_kv: int
    sums: Dict[str, Dict[DirKey, torch.Tensor]] = field(
        default_factory=lambda: {"control": {}, "attack": {}}
    )
    counts: Dict[str, Dict[DirKey, int]] = field(
        default_factory=lambda: {"control": {}, "attack": {}}
    )

    def _add(self, group: str, key: DirKey, vec: torch.Tensor) -> None:
        sums = self.sums[group]
        counts = self.counts[group]
        if key not in sums:
            sums[key] = vec.clone().double()
            counts[key] = 1
        else:
            sums[key] += vec.double()
            counts[key] += 1

    def add_example(
        self,
        group: str,
        whole_vector_cache: Dict[Tuple[str, int], torch.Tensor],
        per_head_cache: Dict[Tuple[str, int], torch.Tensor],
        attention_mask: torch.Tensor,
    ) -> None:
        for (component, layer), tensor in whole_vector_cache.items():
            pooled = pool_activation(component, tensor, attention_mask)
            self._add(group, ("whole_vector", component, layer, None), pooled)

        for (component, layer), tensor in per_head_cache.items():
            pooled = pool_activation(component, tensor, attention_mask)  # (inner_dim,)
            for h in range(self.n_heads):
                head_vec = pooled[h * self.d_kv:(h + 1) * self.d_kv]
                self._add(group, ("per_head", component, layer, h), head_vec)

    def compute_directions(self) -> Dict[DirKey, torch.Tensor]:
        """direction[key] = mean(attack) - mean(control), for keys present in both."""
        directions: Dict[DirKey, torch.Tensor] = {}
        common_keys = set(self.sums["control"]) & set(self.sums["attack"])
        for key in common_keys:
            mean_control = self.sums["control"][key] / self.counts["control"][key]
            mean_attack = self.sums["attack"][key] / self.counts["attack"][key]
            directions[key] = (mean_attack - mean_control).float()
        return directions

    def norm_rows(self, directions: Dict[DirKey, torch.Tensor], tier: str) -> List[dict]:
        rows = []
        for (granularity, component, layer, head_idx), vec in directions.items():
            rows.append({
                "tier": tier,
                "granularity": granularity,
                "component": component,
                "layer": layer,
                "head_idx": head_idx if head_idx is not None else "",
                "direction_norm": float(torch.linalg.norm(vec).item()),
                "n_control": self.counts["control"][(granularity, component, layer, head_idx)],
                "n_attack": self.counts["attack"][(granularity, component, layer, head_idx)],
            })
        return rows


def new_accumulator(model) -> DirectionAccumulator:
    n_heads, d_kv, _ = head_geometry(model)
    return DirectionAccumulator(n_heads=n_heads, d_kv=d_kv)
