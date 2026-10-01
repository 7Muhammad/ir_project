"""
exp18lib/repr_change.py
========================
Experiment 18: full-population ENCODER REPRESENTATION CHANGE, padded control -> attacked input.

Population    = the Exp 16 stage-17 paired manifest (every aligned judged DL19 pair, qrel 2/3 or 0, x 105 attacks)
Scores        = Exp 16 stage 18 (score_control / score_attack / delta_score; successful <=> delta_score > 0)
Representation= the FULL encoder residual stream h in R^(tokens x 768), fp32, one state per encoder layer.

States (hook points, see exp16lib.checkpoints for the transformers 5.9.0 source walk):
  "embedding"   forward-PRE-hook on encoder.block[0]  (= HF encoder hidden_states[0]); reference only
  "L00".."L11"  output of encoder.block[L] (= T5LayerFF residual output, exp16lib.checkpoints
                L{L}_post_mlp). For L <= 10 this is HF hidden_states[L+1]. For L = 11 it is the block
                output BEFORE encoder.final_layer_norm, which HF never returns (HF hidden_states[12]
                is the post-final-norm tensor).
  "final_norm"  encoder.final_layer_norm output = HF last_hidden_state = hidden_states[12] (what the
                decoder cross-attends to); reference only
The main analyses use the 12 layer states L00-L11 only; embedding / final_norm are separate rows.

Regions (positions in the common control/attack layout of length n; A = inserted positions;
P = sorted positions NOT in A. Removing A from the attacked prompt gives the clean prompt exactly
(exp16lib.inputs invariant), so P[k] is the position of clean-prompt token k in BOTH sequences):
  query               P[q0:q1] = the query TEXT tokens (template "Query:" etc. excluded; DECISIONS 4)
  orig_doc            P[d0c:d1c] = exactly the clean passage tokens (injected tokens excluded). In the
                      1,270 "boundary shift" instances (DECISIONS 40) this differs from the stage-18
                      control doc mask by the template ':' that Exp 01 labels as inserted.
  whole_shared        P = every real token present in both sequences: query text + passage + all prompt
                      template tokens ("Query:", "Document:", "Relevant:", </s>)
  whole_full_visible  control: P (its every visible token, == whole_shared);  attack: all n tokens
                      INCLUDING the injected ones. Composition/length confounded; never token-wise.

Metrics (x = control, y = attack):
  one_minus_cosine    0.5 * || x/|x| - y/|y| ||^2   (== 1 - cos, computed without cancellation), float64
  normalized_l2       ||y - x|| / max(||x||, EPS)
  token-wise          the same two per aligned token (query / orig_doc / whole_shared only), summarised
                      per instance by mean and quantiles
  linear CKA          population level (see linear_cka / cka_from_stats), raw and query-centred
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

import exp18lib  # noqa: F401
from exp16lib.inputs import TAIL_LEN, EncodedSeq

STATES: List[str] = ["embedding"] + [f"L{l:02d}" for l in range(12)] + ["final_norm"]
LAYER_STATES: List[str] = [f"L{l:02d}" for l in range(12)]
STATE_LAYER = {s: (-1 if s == "embedding" else 12 if s == "final_norm" else int(s[1:])) for s in STATES}
REGIONS: List[str] = ["query", "orig_doc", "whole_shared", "whole_full_visible"]
TOKENWISE_REGIONS: List[str] = ["query", "orig_doc", "whole_shared"]
N_TW = len(TOKENWISE_REGIONS)
# pooled vectors: control needs 3 (its full-visible pool == whole_shared), attack 4
CTL_OF_REGION = [0, 1, 2, 2]          # region r -> control vector index
ATK_OF_REGION = [0, 1, 2, 3]          # region r -> attack vector index
EPS = 1e-6
QUANTILES = (0.1, 0.5, 0.9)

SUCCESS_GROUPS = ["successful", "unsuccessful", "all"]
RELEVANCE_GROUPS = ["all", "relevant", "nonrelevant"]
RELEVANT, NONRELEVANT = "relevant", "nonrelevant"


# ---- regions --------------------------------------------------------------------------------

@dataclass
class PairRegions:
    n: int
    inserted: List[int]
    query: List[int]
    orig_doc: List[int]
    shared: List[int]


def pair_regions(n: int, inserted: Sequence[int], query_span: Sequence[int], n_passage: int) -> PairRegions:
    """
    Region positions of one control/attack pair (pure; see module docstring).
    n = attacked length, inserted = A, query_span = (q0, q1) in the attacked layout (== clean layout,
    A lies after the query), n_passage = clean passage token count. The clean prompt has n - |A|
    tokens and ends with the TAIL_LEN template tokens, so its passage is [n_clean-4-n_passage, n_clean-4).
    """
    A = sorted(int(i) for i in inserted)
    Aset = set(A)
    if len(Aset) != len(A) or (A and (A[0] < 0 or A[-1] >= n)):
        raise ValueError("inserted positions must be unique and inside the sequence")
    P = [i for i in range(n) if i not in Aset]
    n_clean = len(P)
    d1c = n_clean - TAIL_LEN
    d0c = d1c - n_passage
    q0, q1 = int(query_span[0]), int(query_span[1])
    if not (0 <= q0 < q1 <= d0c) or n_passage < 1:
        raise ValueError(f"bad layout: query ({q0},{q1}), clean doc ({d0c},{d1c})")
    if A and A[0] < q1:
        raise ValueError("inserted position inside the query")
    return PairRegions(n, A, P[q0:q1], P[d0c:d1c], P)


def regions_from_info(info: dict) -> PairRegions:
    return pair_regions(info["seq_len"], info["inserted_positions"], info["query_span"], info["n_passage_tokens"])


def check_pair(atk: EncodedSeq, ctl: EncodedSeq, reg: PairRegions, clean_ids: Sequence[int] = None) -> None:
    """Data-integrity checks of one aligned pair (raise on any failure; never silently drop)."""
    n = reg.n
    if atk.seq_len != n or ctl.seq_len != n:
        raise AssertionError("length mismatch")
    for i in reg.shared:
        if atk.input_ids[i] != ctl.input_ids[i] or ctl.attention_mask[i] != 1:
            raise AssertionError(f"shared position {i}: token ids / control mask differ")
    for i in reg.inserted:
        if ctl.attention_mask[i] != 0 or atk.attention_mask[i] != 1:
            raise AssertionError(f"inserted position {i} not masked in control / visible in attack")
    ins = set(reg.inserted)
    if ins & set(reg.orig_doc) or ins & set(reg.query) or set(reg.query) & set(reg.orig_doc):
        raise AssertionError("region overlap (query / orig_doc / inserted)")
    if [i for i in range(n) if atk.query_mask[i]] != reg.query:
        raise AssertionError("query region differs from the exp16lib query mask")
    if clean_ids is not None:
        if [atk.input_ids[i] for i in reg.shared] != list(clean_ids):
            raise AssertionError("shared tokens != clean prompt")


# ---- batch tensors --------------------------------------------------------------------------

def collate_pairs(pairs: List[Tuple[EncodedSeq, EncodedSeq, PairRegions]], pad_id: int, device) -> Dict[str, torch.Tensor]:
    """
    Rows 0..B-1 = controls, rows B..2B-1 = the matching attacks (same batch -> identical padding and
    kernel shapes for both members of a pair). region_mask [B, S, 3] = (query, orig_doc, shared),
    identical positions for control and attack; attack_full_mask [B, S] = attack attention mask.
    """
    B = len(pairs)
    S = max(r.n for _, _, r in pairs)
    ids = torch.full((2 * B, S), pad_id, dtype=torch.long)
    att = torch.zeros((2 * B, S), dtype=torch.long)
    reg = torch.zeros((B, S, N_TW), dtype=torch.float32)
    for b, (atk, ctl, r) in enumerate(pairs):
        ids[b, :r.n] = torch.tensor(ctl.input_ids)
        att[b, :r.n] = torch.tensor(ctl.attention_mask)
        ids[B + b, :r.n] = torch.tensor(atk.input_ids)
        att[B + b, :r.n] = torch.tensor(atk.attention_mask)
        for k, pos in enumerate((r.query, r.orig_doc, r.shared)):
            reg[b, pos, k] = 1.0
    return {"input_ids": ids.to(device), "attention_mask": att.to(device), "region_mask": reg.to(device),
            "attack_full_mask": att[B:].to(torch.float32).to(device)}


def check_batch(batch: Dict[str, torch.Tensor]) -> None:
    """Tensor-level integrity: shared tokens identical, regions inside the attention of BOTH rows, no padding."""
    B = batch["region_mask"].shape[0]
    ids, att, reg = batch["input_ids"], batch["attention_mask"], batch["region_mask"].bool()
    shared = reg[..., 2]
    if not bool((att[:B].bool() == shared).all()):
        raise AssertionError("control visible tokens != shared region")
    if bool((reg.any(-1) & ~att[B:].bool()).any()):
        raise AssertionError("region position outside the attack attention mask")
    if bool(((ids[:B] != ids[B:]) & shared).any()):
        raise AssertionError("token ids differ at a shared position")
    if bool((reg[..., 0] & reg[..., 1]).any()) or bool((reg[..., :2].any(-1) & ~shared).any()):
        raise AssertionError("query/orig_doc overlap or outside shared")


# ---- metrics --------------------------------------------------------------------------------

def masked_mean(h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """h [B, S, d], mask [B, S, R] (0/1) -> [B, R, d]; every (row, region) must select >= 1 token."""
    cnt = mask.sum(1)
    if bool((cnt <= 0).any()):
        raise ValueError("empty region")
    return torch.einsum("bsr,bsd->brd", mask.to(h.dtype), h) / cnt.to(h.dtype).unsqueeze(-1)


def one_minus_cosine(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Row-wise 1 - cos(x, y) = 0.5 ||x/|x| - y/|y|||^2 (last dim), in the inputs' dtype."""
    xn = x / x.norm(dim=-1, keepdim=True).clamp_min(1e-30)
    yn = y / y.norm(dim=-1, keepdim=True).clamp_min(1e-30)
    return 0.5 * (xn - yn).pow(2).sum(-1)


def normalized_l2(x: torch.Tensor, y: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    """||y - x|| / max(||x||, eps) (x = control = reference)."""
    return (y - x).norm(dim=-1) / x.norm(dim=-1).clamp_min(eps)


def masked_quantiles(v: torch.Tensor, mask: torch.Tensor, qs=QUANTILES) -> torch.Tensor:
    """v [B, S], mask [B, S] bool -> [len(qs), B] (linear interpolation over the selected tokens)."""
    x = torch.where(mask, v, torch.full_like(v, float("nan")))
    return torch.nanquantile(x, torch.tensor(qs, dtype=v.dtype, device=v.device), dim=1)


def state_metrics(h: torch.Tensor, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    One encoder state h [2B, S, d] (controls then attacks) -> per-pair metrics [B, 4 regions] (+ token-wise
    [B, 3]) and the pooled vectors X [B, 3, d] (control) / Y [B, 4, d] (attack), float64.
    """
    B = batch["region_mask"].shape[0]
    hc, ha = h[:B].float(), h[B:].float()
    reg = batch["region_mask"]
    X = masked_mean(hc, reg).double()                                               # [B, 3, d]
    Ya = masked_mean(ha, reg).double()
    Yf = masked_mean(ha, batch["attack_full_mask"].unsqueeze(-1)).double()          # [B, 1, d]
    Y = torch.cat([Ya, Yf], 1)                                                      # [B, 4, d]
    Xr = X[:, CTL_OF_REGION]
    out = {"X": X, "Y": Y, "one_minus_cosine": one_minus_cosine(Xr, Y), "normalized_l2": normalized_l2(Xr, Y),
           "ctl_norm_min": X.norm(dim=-1).min()}
    # token-wise on aligned positions (fp32 tokens, stable 1-cos)
    t_omc = one_minus_cosine(hc, ha)                                                # [B, S]
    cn = hc.norm(dim=-1)
    t_nl2 = (ha - hc).norm(dim=-1) / cn.clamp_min(EPS)
    regb = reg.bool()
    tok_small = int(((cn < EPS) & regb.any(-1)).sum())
    cnt = reg.sum(1)                                                                # [B, 3]
    tw = {"tw_mean_omc": [], "tw_q50_omc": [], "tw_q90_omc": [], "tw_mean_nl2": [], "tw_q50_nl2": [], "tw_q90_nl2": []}
    for k in range(N_TW):
        m = regb[..., k]
        tw["tw_mean_omc"].append((t_omc * m).sum(1) / cnt[:, k])
        tw["tw_mean_nl2"].append((t_nl2 * m).sum(1) / cnt[:, k])
        qo = masked_quantiles(t_omc, m)
        ql = masked_quantiles(t_nl2, m)
        tw["tw_q50_omc"].append(qo[1])
        tw["tw_q90_omc"].append(qo[2])
        tw["tw_q50_nl2"].append(ql[1])
        tw["tw_q90_nl2"].append(ql[2])
    for k, v in tw.items():
        out[k] = torch.stack(v, 1)                                                  # [B, 3]
    out["n_tok_small_norm"] = tok_small
    return out


class StateCapture:
    """
    Hooks on the 14 states (see module docstring). Each hook reduces its [2B, S, d] state immediately with
    state_metrics and hands the result to `sink(state_index, metrics)`; no hidden state is kept.
    """

    def __init__(self, encoder, sink):
        self.encoder, self.sink = encoder, sink
        self.batch = None
        self._h = []
        self.seen = set()

    def set_batch(self, batch):
        self.batch = batch
        self.seen = set()

    def _rec(self, i, h):
        if h.dim() != 3 or h.shape[0] != 2 * self.batch["region_mask"].shape[0]:
            raise RuntimeError(f"state {STATES[i]}: unexpected shape {tuple(h.shape)}")
        self.seen.add(i)
        self.sink(i, state_metrics(h, self.batch))

    def __enter__(self):
        enc = self.encoder

        def pre(module, args, kwargs):
            self._rec(0, args[0] if args else kwargs["hidden_states"])
        self._h.append(enc.block[0].register_forward_pre_hook(pre, with_kwargs=True))
        for L, block in enumerate(enc.block):
            def post(module, args, output, _i=1 + L):
                self._rec(_i, output)
            self._h.append(block.layer[-1].register_forward_hook(post))

        def fin(module, args, output):
            self._rec(len(STATES) - 1, output)
        self._h.append(enc.final_layer_norm.register_forward_hook(fin))
        return self

    def __exit__(self, *exc):
        for h in self._h:
            h.remove()
        self._h = []
        return False


# ---- linear CKA -----------------------------------------------------------------------------

def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Linear CKA (Kornblith et al. 2019, feature-space form), rows = matched instances:
        Xc, Yc = column-centred X, Y
        CKA = ||Yc^T Xc||_F^2 / (||Xc^T Xc||_F * ||Yc^T Yc||_F)
    (== HSIC(K, L) / sqrt(HSIC(K, K) HSIC(L, L)) with linear kernels; the 1/(n-1)^2 factors cancel.)
    """
    X = np.asarray(X, np.float64)
    Y = np.asarray(Y, np.float64)
    Xc, Yc = X - X.mean(0), Y - Y.mean(0)
    num = np.linalg.norm(Yc.T @ Xc, "fro") ** 2
    den = np.linalg.norm(Xc.T @ Xc, "fro") * np.linalg.norm(Yc.T @ Yc, "fro")
    return float(num / den) if den > 0 else float("nan")


def query_center(X: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Subtract each group's (query's) own mean from its rows. Applied to control and attack SEPARATELY."""
    X = np.asarray(X, np.float64).copy()
    g = np.asarray(groups)
    for q in np.unique(g):
        m = g == q
        X[m] -= X[m].mean(0)
    return X


def cka_from_stats(Gxx, Gyy, Gxy, sx, sy, n) -> float:
    """
    Streaming linear CKA. G.. = uncentred second moments (sum_i x_i x_i^T etc.), s. = column sums, n = rows.
    If sx, sy are [Q, d] per-group sums and n is [Q] per-group counts, the centring is WITHIN groups:
        C_xy = sum_i x_i y_i^T - sum_q s_xq s_yq^T / n_q          (groups with n_q = 0 skipped)
    which equals the Gram of the within-query-centred matrices (query_center). A group with n_q = 1
    contributes exactly zero after centring.
    """
    def cen(G, a, b, cnt):
        a, b, cnt = np.atleast_2d(a), np.atleast_2d(b), np.atleast_1d(np.asarray(cnt, np.float64))
        k = cnt > 0
        return G - (a[k] / cnt[k][:, None]).T @ b[k]
    Cxx, Cyy, Cxy = cen(Gxx, sx, sx, n), cen(Gyy, sy, sy, n), cen(Gxy, sx, sy, n)
    den = np.linalg.norm(Cxx, "fro") * np.linalg.norm(Cyy, "fro")
    return float(np.linalg.norm(Cxy, "fro") ** 2 / den) if den > 0 else float("nan")


class CKAAccumulator:
    """
    Sufficient statistics of linear CKA for every (state, region) and every base cell
    cell = 2 * successful + relevant (4 cells), accumulated in float64 on `device`:
        Gxx [S, 3, 4, d, d]  control Grams (query, orig_doc, whole_shared)
        Gyy [S, 4, 4, d, d]  attack Grams (query, orig_doc, whole_shared, whole_full_visible)
        Gxy [S, 4, 4, d, d]  control-attack cross moments per region (control full-visible = whole_shared)
        sx [S, 3, 4, Q, d], sy [S, 4, 4, Q, d] per-query sums; counts [4, Q]
    Any population that is a union of cells (success x relevance groups) is then exact by summation.
    """

    def __init__(self, n_states: int, n_queries: int, d: int, device):
        f = dict(dtype=torch.float64, device=device)
        self.Gxx = torch.zeros(n_states, 3, 4, d, d, **f)
        self.Gyy = torch.zeros(n_states, 4, 4, d, d, **f)
        self.Gxy = torch.zeros(n_states, 4, 4, d, d, **f)
        self.sx = torch.zeros(n_states, 3, 4, n_queries, d, **f)
        self.sy = torch.zeros(n_states, 4, 4, n_queries, d, **f)
        self.counts = torch.zeros(4, n_queries, **f)

    def add_counts(self, cell: torch.Tensor, qidx: torch.Tensor):
        self.counts.index_put_((cell, qidx), torch.ones_like(cell, dtype=torch.float64), accumulate=True)

    def update(self, s: int, X: torch.Tensor, Y: torch.Tensor, cell: torch.Tensor, qidx: torch.Tensor):
        """X [B, 3, d], Y [B, 4, d] float64; cell, qidx [B] long (same device)."""
        Xr = X[:, CTL_OF_REGION]
        for c in torch.unique(cell).tolist():
            m = cell == c
            xc, yc, xr = X[m].transpose(0, 1), Y[m].transpose(0, 1), Xr[m].transpose(0, 1)   # [R, b, d]
            self.Gxx[s, :, c] += xc.transpose(1, 2) @ xc
            self.Gyy[s, :, c] += yc.transpose(1, 2) @ yc
            self.Gxy[s, :, c] += xr.transpose(1, 2) @ yc
            self.sx[s, :, c].index_add_(1, qidx[m], xc)
            self.sy[s, :, c].index_add_(1, qidx[m], yc)

    def cka_table(self, cells: Sequence[int]) -> List[Dict]:
        """CKA raw + query-centred for every (state, region) of the population = union of `cells`."""
        cells = list(cells)
        cnt = self.counts[cells].sum(0).cpu().numpy()                  # [Q]
        rows = []
        for s in range(self.Gxx.shape[0]):
            for r in range(len(REGIONS)):
                ci, ai = CTL_OF_REGION[r], ATK_OF_REGION[r]
                Gxx = self.Gxx[s, ci, cells].sum(0).cpu().numpy()
                Gyy = self.Gyy[s, ai, cells].sum(0).cpu().numpy()
                Gxy = self.Gxy[s, r, cells].sum(0).cpu().numpy()
                sxq = self.sx[s, ci, cells].sum(0).cpu().numpy()         # [Q, d]
                syq = self.sy[s, ai, cells].sum(0).cpu().numpy()
                n = cnt.sum()
                raw = cka_from_stats(Gxx, Gyy, Gxy, sxq.sum(0), syq.sum(0), n) if n >= 2 else float("nan")
                qc = cka_from_stats(Gxx, Gyy, Gxy, sxq, syq, cnt) if n >= 2 else float("nan")
                rows.append({"state": STATES[s], "layer": STATE_LAYER[STATES[s]], "region": REGIONS[r],
                             "cka_raw": raw, "cka_query_centered": qc, "n_instances": int(n),
                             "n_queries": int((cnt > 0).sum()), "n_queries_singleton": int((cnt == 1).sum()),
                             "n_instances_query_centered": int(cnt[cnt >= 2].sum())})
        return rows


def cells_of(success_group: str, relevance_group: str) -> List[int]:
    """Base cells (2 * successful + relevant) of a (success group, relevance group) population."""
    ss = {"successful": [1], "unsuccessful": [0], "all": [0, 1]}[success_group]
    rr = {"relevant": [1], "nonrelevant": [0], "all": [0, 1]}[relevance_group]
    return [2 * s + r for s in ss for r in rr]


def group_mask(successful: np.ndarray, relevance: np.ndarray, success_group: str, relevance_group: str) -> np.ndarray:
    """Row filter for a (success group, relevance group) population (the same definition as cells_of)."""
    successful = np.asarray(successful, bool)
    relevance = np.asarray(relevance)
    m = np.ones(len(successful), bool)
    if success_group == "successful":
        m &= successful
    elif success_group == "unsuccessful":
        m &= ~successful
    elif success_group != "all":
        raise ValueError(success_group)
    if relevance_group in (RELEVANT, NONRELEVANT):
        m &= relevance == relevance_group
    elif relevance_group != "all":
        raise ValueError(relevance_group)
    return m


# ---- query-level bootstrap ------------------------------------------------------------------

def query_bootstrap_mean(sums: np.ndarray, counts: np.ndarray, n_boot: int, seed: int, alpha: float = 0.05):
    """
    Instance-weighted mean with a query (cluster) bootstrap. sums [Q, ...] per-query sums, counts [Q].
    Resample queries with replacement; mean_b = sum_q w_bq S_q / sum_q w_bq n_q. Returns (mean, lo, hi).
    """
    sums = np.asarray(sums, np.float64)
    counts = np.asarray(counts, np.float64)
    keep = counts > 0
    sums, counts = sums[keep], counts[keep]
    Q = len(counts)
    mean = sums.sum(0) / counts.sum()
    if Q < 2:
        nan = np.full_like(mean, np.nan)
        return mean, nan, nan
    rng = np.random.default_rng(seed)
    W = rng.multinomial(Q, np.full(Q, 1.0 / Q), size=n_boot).astype(np.float64)   # [n_boot, Q]
    bs = np.tensordot(W, sums, axes=(1, 0)) / (W @ counts).reshape((-1,) + (1,) * (sums.ndim - 1))
    lo, hi = np.nanpercentile(bs, [100 * alpha / 2, 100 * (1 - alpha / 2)], axis=0)
    return mean, lo, hi
