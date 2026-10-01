"""
exp19lib/head_change.py
========================
Per-head control -> attack change of the pre-o_proj head-output representation (Exp 19).

Hook (identical to exp16lib.heads.EncoderHeadCapture): forward-PRE-hook on
encoder.block[L].layer[0].SelfAttention.o; args[0] is [2B, S, 12 * 64] = concat_h(head_h) (T5
unshape: [B, H, S, d_kv] -> transpose -> view, so head h = slice [h*64, (h+1)*64)). Rows 0..B-1 are
padded controls, B..2B-1 the matching attacks (exp18lib.repr_change.collate_pairs).

Regions (exp18lib.repr_change.pair_regions, the exact Exp 18 masks): region_mask[..., 0] = query text,
region_mask[..., 1] = original document (clean passage tokens only; inserted + padding excluded).

Per (instance, layer, head, region):
  one_minus_cosine / normalized_l2       of the mean-pooled 64-d vectors (float64)
  tw_mean_omc / tw_mean_nl2              mean over the region's aligned tokens of per-token 1 - cos / L2
Population level: linear CKA (raw + query-centred) from streamed float64 64x64 Gram statistics.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import torch

import exp19lib  # noqa: F401
from exp18lib.repr_change import EPS, cka_from_stats, normalized_l2, one_minus_cosine

N_LAYERS, N_HEADS, D_KV = 12, 12, 64
REGIONS = ["query", "orig_doc"]
N_REG = len(REGIONS)
METRICS = ["one_minus_cosine", "normalized_l2", "tw_mean_omc", "tw_mean_nl2"]
HEAD_LABELS = [f"L{l}H{h}" for l in range(N_LAYERS) for h in range(N_HEADS)]


def split_heads(inner: torch.Tensor, n_heads: int = N_HEADS) -> torch.Tensor:
    """[B, S, H * d] pre-o_proj input -> [B, S, H, d] (head h = slice [h*d, (h+1)*d), EncoderHeadCapture)."""
    B, S, D = inner.shape
    return inner.reshape(B, S, n_heads, D // n_heads)


def head_metrics(inner: torch.Tensor, region_mask: torch.Tensor, n_heads: int = N_HEADS) -> Dict[str, torch.Tensor]:
    """
    inner [2B, S, H*d] (controls then attacks), region_mask [B, S, >=2] (query, orig_doc, ...).
    -> X, Y [B, 2, H, d] float64 pooled control / attack, metrics [B, 2, H], diagnostics.
    """
    B = region_mask.shape[0]
    h = split_heads(inner.float(), n_heads)
    hc, ha = h[:B], h[B:]
    reg = region_mask[..., :N_REG].float()
    cnt = reg.sum(1)                                                          # [B, 2]
    if bool((cnt <= 0).any()):
        raise ValueError("empty query / orig_doc region")
    X = (torch.einsum("bsr,bshd->brhd", reg, hc) / cnt[:, :, None, None]).double()
    Y = (torch.einsum("bsr,bshd->brhd", reg, ha) / cnt[:, :, None, None]).double()
    t_omc = one_minus_cosine(hc, ha)                                          # [B, S, H]
    cn = hc.norm(dim=-1)
    t_nl2 = (ha - hc).norm(dim=-1) / cn.clamp_min(EPS)
    small = (cn < EPS) & reg.bool().any(-1, keepdim=True)
    return {"X": X, "Y": Y,
            "one_minus_cosine": one_minus_cosine(X, Y), "normalized_l2": normalized_l2(X, Y),
            "tw_mean_omc": torch.einsum("bsr,bsh->brh", reg, t_omc) / cnt[:, :, None],
            "tw_mean_nl2": torch.einsum("bsr,bsh->brh", reg, t_nl2) / cnt[:, :, None],
            "n_tok_small_norm": int(small.sum()), "pooled_ctl_norm_min": float(X.norm(dim=-1).min())}


class HeadCapture:
    """Pre-hooks on SelfAttention.o of every encoder layer; each hook reduces immediately via head_metrics -> sink(L, m)."""

    def __init__(self, encoder, sink, keep_inputs: bool = False):
        self.encoder, self.sink, self.keep_inputs = encoder, sink, keep_inputs
        self.region_mask = None
        self.inputs: Dict[int, torch.Tensor] = {}
        self.seen = set()
        self._h = []

    def set_batch(self, region_mask):
        self.region_mask, self.inputs, self.seen = region_mask, {}, set()

    def _hook(self, L):
        def fn(module, args):
            inner = args[0]
            if inner.dim() != 3 or inner.shape[-1] != self.encoder.config.num_heads * self.encoder.config.d_kv:
                raise RuntimeError(f"layer {L}: unexpected o_proj input {tuple(inner.shape)}")
            if self.keep_inputs:
                self.inputs[L] = inner.detach().clone()
            self.seen.add(L)
            self.sink(L, head_metrics(inner, self.region_mask, self.encoder.config.num_heads))
        return fn

    def __enter__(self):
        for L, block in enumerate(self.encoder.block):
            self._h.append(block.layer[0].SelfAttention.o.register_forward_pre_hook(self._hook(L)))
        return self

    def __exit__(self, *exc):
        for h in self._h:
            h.remove()
        self._h = []
        return False


class HeadCKAAccumulator:
    """
    Linear-CKA sufficient statistics per (layer, region, cell, head), cell = 2 * successful + relevant:
      Gxx, Gyy, Gxy [L, 2, 4, H, d, d] float64;  sx, sy [L, 2, 4, Q, H, d];  counts [4, Q]
    (same estimator and query-centring convention as exp18lib.repr_change.CKAAccumulator).
    """

    def __init__(self, n_layers: int, n_heads: int, d: int, n_queries: int, device):
        f = dict(dtype=torch.float64, device=device)
        self.Gxx = torch.zeros(n_layers, N_REG, 4, n_heads, d, d, **f)
        self.Gyy = torch.zeros_like(self.Gxx)
        self.Gxy = torch.zeros_like(self.Gxx)
        self.sx = torch.zeros(n_layers, N_REG, 4, n_queries, n_heads, d, **f)
        self.sy = torch.zeros_like(self.sx)
        self.counts = torch.zeros(4, n_queries, **f)

    def add_counts(self, cell, qidx):
        self.counts.index_put_((cell, qidx), torch.ones_like(cell, dtype=torch.float64), accumulate=True)

    def update(self, L: int, X: torch.Tensor, Y: torch.Tensor, cell: torch.Tensor, qidx: torch.Tensor):
        """X, Y [B, 2, H, d] float64."""
        for c in torch.unique(cell).tolist():
            m = cell == c
            x, y, q = X[m], Y[m], qidx[m]
            self.Gxx[L, :, c] += torch.einsum("brhd,brhe->rhde", x, x)
            self.Gyy[L, :, c] += torch.einsum("brhd,brhe->rhde", y, y)
            self.Gxy[L, :, c] += torch.einsum("brhd,brhe->rhde", x, y)
            self.sx[L, :, c].index_add_(1, q, x.transpose(0, 1))
            self.sy[L, :, c].index_add_(1, q, y.transpose(0, 1))

    def state(self) -> Dict[str, np.ndarray]:
        return {k: getattr(self, k).cpu().numpy() for k in ("Gxx", "Gyy", "Gxy", "sx", "sy", "counts")}


def cka_table_from_state(st: Dict[str, np.ndarray], cells: Sequence[int]) -> List[Dict]:
    """CKA raw + query-centred per (layer, head, region) for the population = union of `cells`."""
    cells = list(cells)
    cnt = st["counts"][cells].sum(0)
    n = cnt.sum()
    rows = []
    nL, _, _, nH = st["Gxx"].shape[:4]
    for L in range(nL):
        for r in range(N_REG):
            Gxx = st["Gxx"][L, r, cells].sum(0)
            Gyy = st["Gyy"][L, r, cells].sum(0)
            Gxy = st["Gxy"][L, r, cells].sum(0)
            sx = st["sx"][L, r, cells].sum(0)                                  # [Q, H, d]
            sy = st["sy"][L, r, cells].sum(0)
            for h in range(nH):
                ok = n >= 2
                raw = cka_from_stats(Gxx[h], Gyy[h], Gxy[h], sx[:, h].sum(0), sy[:, h].sum(0), n) if ok else np.nan
                qc = cka_from_stats(Gxx[h], Gyy[h], Gxy[h], sx[:, h], sy[:, h], cnt) if ok else np.nan
                rows.append({"layer": L, "head": h, "head_label": f"L{L}H{h}", "region": REGIONS[r],
                             "cka_raw": raw, "cka_query_centered": qc, "n_instances": int(n),
                             "n_queries": int((cnt > 0).sum()), "n_queries_singleton": int((cnt == 1).sum())})
    return rows


# ---- causal-head comparison helpers -------------------------------------------------------------

def topk_enrichment(values: np.ndarray, is_causal: np.ndarray, ks=(5, 10, 18)) -> List[Dict]:
    """Causal heads among the top-k (largest values). Expected = k * n_causal / n; hypergeometric P(X >= obs)."""
    from scipy.stats import hypergeom
    values = np.asarray(values, float)
    is_causal = np.asarray(is_causal, bool)
    order = np.argsort(-values, kind="stable")
    N, K = len(values), int(is_causal.sum())
    out = []
    for k in ks:
        obs = int(is_causal[order[:k]].sum())
        out.append({"k": k, "n_causal_in_topk": obs, "expected": k * K / N,
                    "enrichment": obs / (k * K / N) if K else np.nan,
                    "p_hypergeom_ge": float(hypergeom.sf(obs - 1, N, K, k))})
    return out


def within_layer_percentile(values: np.ndarray, layers: np.ndarray) -> np.ndarray:
    """Percentile rank (0 = smallest, 1 = largest) of each head among the heads of its own layer."""
    from scipy.stats import rankdata
    values, layers = np.asarray(values, float), np.asarray(layers)
    out = np.empty(len(values))
    for L in np.unique(layers):
        m = layers == L
        out[m] = (rankdata(values[m]) - 1) / max(m.sum() - 1, 1)
    return out


def layer_stratified_permutation(values: np.ndarray, layers: np.ndarray, is_causal: np.ndarray,
                                 n_perm: int = 20000, seed: int = 42):
    """
    Mean within-layer percentile of the causal heads vs a null that re-draws, IN EACH LAYER, the same number of
    'causal' heads at random (controls for layer-level differences in change magnitude). Returns (obs, p_ge, null mean).
    """
    pct = within_layer_percentile(values, layers)
    is_causal = np.asarray(is_causal, bool)
    obs = float(pct[is_causal].mean())
    rng = np.random.default_rng(seed)
    idx_by_layer = [(np.flatnonzero(layers == L), int(is_causal[layers == L].sum())) for L in np.unique(layers)]
    null = np.empty(n_perm)
    for i in range(n_perm):
        s, c = 0.0, 0
        for idx, k in idx_by_layer:
            if k:
                s += pct[rng.choice(idx, k, replace=False)].sum()
                c += k
        null[i] = s / c
    return obs, float((null >= obs - 1e-12).mean()), float(null.mean())
