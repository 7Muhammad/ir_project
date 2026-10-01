"""
exp17lib/attention_stats.py
============================
Attention-DISTRIBUTION statistics of encoder self-attention, reduced per source row and then
averaged over a source region; attention matrices are never stored.

Notation: A[b, h, i, j] = attention of source/query position i to key/target position j
(post-softmax, as returned by the eager T5 attention; masked keys already carry weight 0).

Key scopes (which keys a row's distribution is taken over; p is RENORMALISED to sum to 1):
  full_visible  all visible keys of the sequence (attention_mask == 1): query, document, template
                tokens, and — in attacked sequences — the injected tokens. Masked control
                insertion slots and batch padding are never visible.
  shared_key    keys visible in BOTH the padded control and the attacked sequence
                = attention_mask & ~inserted_mask. For clean / control sequences this equals
                full_visible (control insertion slots are already masked); for attacked
                sequences it removes the injected tokens as keys.

Per-row metrics (N = number of allowed keys, p renormalised over them):
  entropy       H = -sum_j p_j log p_j                     (nats; diagnostic)
  entropy_norm  H / log N                                  primary; NaN if N <= 1
  max_attn      max_j p_j                                  (= top-1 mass)
  top3, top5    sum of the k largest p_j, k_eff = min(k, N)
  neff          exp(H)                                     effective number of attended keys
  neff_norm     exp(H) / N                                 in [1/N, 1]
  dist          sum_j p_j |i - j|                          token positions (absolute, gaps kept)
  dist_norm     dist / max(1, n_tokens - 1)                n_tokens = unpadded sequence length
  local_mass    sum_{|i-j| <= window} p_j                  window = 5 (single locality window)
Rows with N == 0 (or zero mass on the allowed keys) give NaN everywhere.

Source regions (masks from exp16lib.metrics_screen.region_masks, never re-derived):
  query  query_mask
  doc    doc_mask                          (attacked: original passage + injected tokens)
  orig   doc_mask & ~inserted_mask         (clean / control: == doc)
  ins    doc_mask &  inserted_mask         (attacked only; empty -> NaN otherwise)
Template tokens are never a source region (they stay keys in full_visible scope).

Region-to-region mass (full_visible scope, attacked sequences; directions MASS_DIRS):
  raw   mean_{i in S} sum_{j in B} p_ij
  norm  raw / (|B| / N_visible)            1 = token-count-uniform baseline
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

METRICS = ["entropy_norm", "entropy", "max_attn", "top3", "top5", "neff", "neff_norm", "dist", "dist_norm",
           "local_mass"]
REGIONS = ["query", "doc", "orig", "ins"]
SCOPES = ["full_visible", "shared_key"]
# (source region, target region); the last two are context rows (original-document tokens as a
# same-sequence, size-normalised reference for the injected tokens)
MASS_DIRS: List[Tuple[str, str]] = [("query", "ins"), ("ins", "query"), ("orig", "ins"), ("ins", "orig"),
                                    ("query", "orig"), ("orig", "query")]
MASS_NAMES = [f"{s}_to_{t}" for s, t in MASS_DIRS]
LOCAL_WINDOW = 5


# ---- reference implementation (NumPy, one row; used by tests and the validation stage) -----------

def row_stats_np(a_row: np.ndarray, key_mask: np.ndarray, i: int, n_tokens: int,
                 window: int = LOCAL_WINDOW) -> Dict[str, float]:
    """Direct NumPy computation of every metric for one source row i (float64)."""
    a_row = np.asarray(a_row, np.float64)
    keys = np.where(np.asarray(key_mask, bool))[0]
    out = {m: np.nan for m in METRICS}
    if len(keys) == 0 or a_row[keys].sum() <= 0:
        return out
    p = a_row[keys] / a_row[keys].sum()
    N = len(keys)
    H = float(-sum(x * np.log(x) for x in p if x > 0))
    srt = np.sort(p)[::-1]
    d = np.abs(keys - i)
    out.update(entropy=H, entropy_norm=H / np.log(N) if N > 1 else np.nan, max_attn=float(srt[0]),
               top3=float(srt[:min(3, N)].sum()), top5=float(srt[:min(5, N)].sum()), neff=float(np.exp(H)),
               neff_norm=float(np.exp(H) / N), dist=float((p * d).sum()),
               dist_norm=float((p * d).sum() / max(1, n_tokens - 1)), local_mass=float(p[d <= window].sum()))
    return out


def region_mass_np(A: np.ndarray, src: np.ndarray, tgt: np.ndarray, visible: np.ndarray) -> Tuple[float, float]:
    """A [S, S] one head -> (raw, normalised) mass from src rows to tgt keys over visible keys (float64)."""
    src, tgt, visible = (np.asarray(x, bool) for x in (src, tgt, visible))
    if not src.any() or not tgt.any():
        return np.nan, np.nan
    P = np.asarray(A, np.float64)[src][:, visible]
    P = P / P.sum(1, keepdims=True)
    raw = float(P[:, tgt[visible]].sum(1).mean())
    return raw, raw / (tgt.sum() / visible.sum())


# ---- batched torch implementation --------------------------------------------------------------

def _dist_matrix(S: int, device) -> torch.Tensor:
    r = torch.arange(S, device=device, dtype=torch.float32)
    return (r[:, None] - r[None, :]).abs()                            # [S, S]


def row_stats(A: torch.Tensor, key_mask: torch.Tensor, n_tokens: torch.Tensor,
              window: int = LOCAL_WINDOW) -> Dict[str, torch.Tensor]:
    """
    A [B, H, S, S] attention (float), key_mask [B, S] bool, n_tokens [B] -> {metric: [B, H, S]} float32.
    The distribution of each row is A restricted to key_mask and renormalised.
    """
    B, Hh, S, _ = A.shape
    K = key_mask[:, None, None, :].to(A.dtype)
    P = A * K
    Z = P.sum(-1, keepdim=True)
    N = key_mask.sum(-1).to(torch.float32)[:, None, None].expand(B, Hh, S)   # [B, H, S]
    ok = (Z[..., 0] > 0) & (N > 0)
    P = P / torch.where(Z > 0, Z, torch.ones_like(Z))
    H = -torch.xlogy(P, P).sum(-1)
    k = min(5, S)
    top = torch.topk(P, k, dim=-1).values.cumsum(-1)                 # zeros beyond N add nothing: k_eff = min(k, N)
    D = _dist_matrix(S, A.device)
    dist = (P * D).sum(-1)
    nan = torch.tensor(float("nan"), device=A.device)
    logN = torch.log(N)
    out = {"entropy": H,
           "entropy_norm": torch.where(N > 1, H / torch.where(N > 1, logN, torch.ones_like(logN)), nan),
           "max_attn": top[..., 0], "top3": top[..., min(3, k) - 1], "top5": top[..., k - 1],
           "neff": torch.exp(H), "neff_norm": torch.exp(H) / N.clamp(min=1),
           "dist": dist,
           "dist_norm": dist / (n_tokens.to(torch.float32) - 1).clamp(min=1)[:, None, None],
           "local_mass": (P * (D <= window).to(P.dtype)).sum(-1)}
    return {m: torch.where(ok, v, nan).float() for m, v in out.items()}


def region_mean(stat: torch.Tensor, region: torch.Tensor) -> torch.Tensor:
    """stat [..., B, H, S], region [B, S] bool -> mean over region rows [..., B, H] (NaN rows skipped; empty -> NaN)."""
    R = region[:, None, :].expand_as(stat) & ~torch.isnan(stat)
    cnt = R.sum(-1)
    s = torch.where(R, stat, torch.zeros_like(stat)).sum(-1)
    return torch.where(cnt > 0, s / cnt.clamp(min=1), torch.full_like(s, float("nan")))


def region_mass(A: torch.Tensor, visible: torch.Tensor, src: torch.Tensor, tgt: torch.Tensor):
    """A [B, H, S, S], masks [B, S] bool -> (raw [B, H], norm [B, H]) with rows renormalised over visible keys."""
    P = A * visible[:, None, None, :].to(A.dtype)
    P = P / P.sum(-1, keepdim=True).clamp(min=1e-30)
    m = (P * tgt[:, None, None, :].to(A.dtype)).sum(-1)                      # [B, H, S]
    raw = region_mean(m, src)
    frac = tgt.sum(-1).float() / visible.sum(-1).float().clamp(min=1)       # [B]
    norm = raw / frac[:, None]
    bad = (tgt.sum(-1) == 0)[:, None].expand_as(raw)
    return torch.where(bad, torch.full_like(raw, float("nan")), raw), torch.where(bad, torch.full_like(norm, float("nan")), norm)


def batch_masks(attention_mask, query_mask, doc_mask, inserted_mask) -> Dict[str, torch.Tensor]:
    """Token masks [B, S] (any int/bool) -> bool masks for keys (per scope) and source regions."""
    vis, q, d, ins = (torch.as_tensor(x).bool() for x in (attention_mask, query_mask, doc_mask, inserted_mask))
    ins_doc = d & ins                                     # real injected tokens (control slots are outside doc_mask)
    return {"full_visible": vis, "shared_key": vis & ~ins,
            "query": q & vis, "doc": d & vis, "orig": d & ~ins & vis, "ins": ins_doc & vis}


@torch.inference_mode()
def batch_statistics(attentions: Sequence[torch.Tensor], masks: Dict[str, torch.Tensor], n_tokens: torch.Tensor,
                     window: int = LOCAL_WINDOW):
    """
    attentions: per layer [B, H, S, S] -> (
        R [B, len(REGIONS), len(SCOPES), len(METRICS), L, H] float32 numpy,
        M [B, len(MASS_DIRS), 2 (raw, norm), L, H] float32 numpy)
    """
    L, (B, Hh, S, _) = len(attentions), attentions[0].shape
    dev = attentions[0].device
    R = torch.full((B, len(REGIONS), len(SCOPES), len(METRICS), L, Hh), float("nan"), device=dev)
    M = torch.full((B, len(MASS_DIRS), 2, L, Hh), float("nan"), device=dev)
    for li, A in enumerate(attentions):
        A = A.float()
        for si, sc in enumerate(SCOPES):
            st = row_stats(A, masks[sc], n_tokens, window)
            stk = torch.stack([st[m] for m in METRICS])                         # [n_metrics, B, H, S]
            for ri, r in enumerate(REGIONS):
                R[:, ri, si, :, li] = region_mean(stk, masks[r]).transpose(0, 1)
            del st, stk
        for di, (s, t) in enumerate(MASS_DIRS):
            raw, norm = region_mass(A, masks["full_visible"], masks[s], masks[t])
            M[:, di, 0, li], M[:, di, 1, li] = raw, norm
    return R.cpu().numpy(), M.cpu().numpy()


# ---- group statistics ----------------------------------------------------------------------------

def residualize(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """OLS residual of y on [1, x] (NaNs in y kept as NaN); used for the length-confound check."""
    out = np.full_like(y, np.nan, dtype=np.float64)
    ok = ~np.isnan(y) & ~np.isnan(x)
    if ok.sum() < 3:
        return out
    X = np.column_stack([np.ones(ok.sum()), x[ok]])
    beta, *_ = np.linalg.lstsq(X, y[ok], rcond=None)
    out[ok] = y[ok] - X @ beta
    return out
