"""
exp16lib/paired.py
===================
Paired extension of Exp 16 (stages 17-21): for the SAME judged DL19 document,
does a successful attack make its late-encoder query/document similarity more
abnormal than its own matched padded control?

  instance      = (qid, docid, attack configuration); one of the 105 grid attacks
  pair          = padded control -> attacked input (exp16lib.inputs.encode_attack_and_control)
  success       = delta_score = score_attack - score_control > 0   (monoT5 logit(true) - logit(false))
  groups        = qrel 2/3 ("relevant") and qrel 0 ("nonrelevant"), never pooled;
                  qrel 1 and unjudged documents excluded (exp16lib.qrels.relevance_group)
  features      = the encoder heads of the configured scope (exp16lib.heads.all_encoder_heads), 64-d head
                  output BEFORE o_proj, cos(mean query-text, mean document) (EncoderHeadCapture):
                  paired.scope "L9_L11" = 36 heads of layers 9-11 (primary), "all_layers" = 144 heads of
                  layers 0-11 (scope extension; identical method, own *_all_layers output directories)
  reference     = per CV fold: padded CONTROLS of successful qrel 2/3 instances of TRAINING queries,
                  pooled over ALL attack configurations (one mu/sigma per head per fold)
  abnormal      = |z| > 2 (fixed)
  folds         = exp16lib.anomaly.query_folds over the union of judged attackable qids, 5 folds, seed 42

Nothing here redefines spans, pooling, hooks, scoring, alignment, folds or the
attack grid; it only combines the existing pieces.
"""

from __future__ import annotations

import hashlib
import pathlib
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import torch

import exp16lib  # noqa: F401
from exp16lib.anomaly import (abnormal_counts, auroc, choose_threshold, fit_reference, head_ranking_from_z,
                              inner_heldout_z, zscores)
from exp16lib.checkpoints import CHECKPOINTS
from exp16lib.engine import forward_batch
from exp16lib.heads import EncoderHeadCapture, all_encoder_heads
from exp16lib.inputs import EncodedSeq, EncodingError, collate, encode_attack_and_control

# ---- locked design ------------------------------------------------------------------------
SCOPES = {"L9_L11": [9, 10, 11], "all_layers": list(range(12))}
DEFAULT_SCOPE = "L9_L11"
SCOPE = DEFAULT_SCOPE
LAYERS = SCOPES[SCOPE]          # set by configure(); read at run time as P.LAYERS / P.N_HEADS
N_HEADS = 12 * len(LAYERS)
K_ABNORMAL = 2.0
N_FOLDS = 5
FOLD_SEED = 42
RELEVANT, NONRELEVANT = "relevant", "nonrelevant"
GROUPS = [RELEVANT, NONRELEVANT]
GROUP_LABEL = {RELEVANT: "relevant base documents (qrel 2/3)", NONRELEVANT: "non-relevant base documents (qrel 0)"}

# ---- output layout (new directories only; old Exp 16 outputs are never written) ------------
MANIFEST_DIR = "17_paired_manifest"
_SCOPED = {"FORWARD_DIR": "18_paired_forward", "ANOMALY_DIR": "19_paired_anomaly",
           "DETECTOR_DIR": "20_paired_detector", "REPORT_DIR": "21_paired_report", "PLOT_DIR": "plots/paired"}
FORWARD_DIR, ANOMALY_DIR, DETECTOR_DIR, REPORT_DIR, PLOT_DIR = _SCOPED.values()


def scope_suffix(scope: str) -> str:
    return "" if scope == DEFAULT_SCOPE else f"_{scope}"


# every paired directory of every scope (excluded from the old-output fingerprint)
PAIRED_OUTPUTS = [MANIFEST_DIR] + [v + scope_suffix(s) for s in SCOPES for v in _SCOPED.values()]

DT = {"qid": str, "docid": str, "pair_id": str, "attack_token": str, "attack_name": str, "relevance_group": str}

INSTANCE_META = ["attack_id", "attack_name", "attack_token", "attack_position", "repetitions", "pair_id", "qid",
                 "docid", "qrel_grade", "relevance_group", "seq_len", "n_inserted", "n_query_tokens",
                 "n_document_tokens_attack", "n_document_tokens_control", "alignment_boundary_shift"]


def heads():
    h = all_encoder_heads(LAYERS)
    assert len(h) == N_HEADS
    return h


def head_labels() -> List[str]:
    return [h.label for h in heads()]


def head_cols(side: str) -> List[str]:
    """side in {'control', 'attack'} -> the scope's head-similarity columns, ordered by (layer, head)."""
    return [f"{side}_head_{l}" for l in head_labels()]


def ckpt_cols(side: str) -> List[str]:
    return [f"{side}_ckpt_{c}" for c in CHECKPOINTS]


def configure(scope: str) -> None:
    """Select the head scope: layers, head count and the scope's output directories (manifest is shared)."""
    global SCOPE, LAYERS, N_HEADS, FORWARD_DIR, ANOMALY_DIR, DETECTOR_DIR, REPORT_DIR, PLOT_DIR
    if scope not in SCOPES:
        raise ValueError(f"paired.scope must be one of {list(SCOPES)} (got {scope})")
    SCOPE, LAYERS = scope, SCOPES[scope]
    N_HEADS = 12 * len(LAYERS)
    FORWARD_DIR, ANOMALY_DIR, DETECTOR_DIR, REPORT_DIR, PLOT_DIR = (v + scope_suffix(scope) for v in _SCOPED.values())


def scope_label() -> str:
    return f"layers {LAYERS[0]}–{LAYERS[-1]} ({N_HEADS} heads)"


def validate_cfg(cfg: dict) -> dict:
    p = cfg["paired"]
    scope = p.get("scope", DEFAULT_SCOPE)
    configure(scope)
    locked = {"success_threshold": 0.0, "abnormal_z": K_ABNORMAL, "n_folds": N_FOLDS, "fold_seed": FOLD_SEED,
              "layers": SCOPES[scope]}
    for k, v in locked.items():
        if p[k] != v:
            raise ValueError(f"paired.{k} is locked to {v} for scope {scope} (got {p[k]})")
    return p


def pair_id(qid, docid) -> str:
    return f"{qid}_{docid}"


def encode_instance(tok, query: str, passage: str, attacked_passage: str, max_len: int):
    """
    encode_attack_and_control, with the stage-00 failure rule: an Exp 01 ALIGNMENT failure returns
    (None, None, reason) so the caller logs and excludes it; any other EncodingError is raised.
    """
    try:
        return encode_attack_and_control(tok, query, passage, attacked_passage, max_len)
    except EncodingError as e:
        if "alignment failed" not in str(e):
            raise
        return None, None, str(e)


# ---- forward: score + 25 checkpoints + scope heads from ONE full forward ----------------------

@torch.inference_mode()
def forward_with_heads(model, batch, hs, true_id: int, false_id: int):
    """engine.forward_batch (score + 25 checkpoint cosines) with the head pre-hooks attached to the same pass."""
    with EncoderHeadCapture(model.encoder, hs) as ec:
        ec.set_masks(batch["query_mask"], batch["doc_mask"])
        sims, scores, _ = forward_batch(model, batch, True, true_id, false_id)
        H = torch.stack([ec.cos[h.label] for h in hs], 1)
    return sims, H, scores


def run_paired_sequences(model, seqs: List[EncodedSeq], pad_id: int, device, batch_size: int, hs,
                         true_id: int, false_id: int):
    """Length-sorted batches (as engine.run_sequences) -> (ckpt [N, 25], heads [N, N_HEADS], score [N]) in input order."""
    for s in seqs:
        if s.n_query_tokens < 1 or s.n_doc_tokens < 1:
            raise ValueError("every sequence needs >= 1 query and >= 1 document token")
    order = sorted(range(len(seqs)), key=lambda i: seqs[i].seq_len)
    C = None                                    # [N, n_checkpoints] (25 for monoT5-base; sized from the capture)
    H = np.full((len(seqs), len(hs)), np.nan)
    S = np.full(len(seqs), np.nan)
    for start in range(0, len(order), batch_size):
        idx = order[start:start + batch_size]
        sims, h, sc = forward_with_heads(model, collate([seqs[i] for i in idx], pad_id, device), hs, true_id, false_id)
        if C is None:
            C = np.full((len(seqs), sims.shape[1]), np.nan)
        C[idx], H[idx], S[idx] = sims.cpu().numpy(), h.cpu().numpy(), sc.cpu().numpy()
    if not (np.isfinite(C).all() and np.isfinite(H).all() and np.isfinite(S).all()):
        raise FloatingPointError("non-finite similarity or score")
    return C, H, S


# ---- I/O -------------------------------------------------------------------------------------

def load_forward_rows(forward_dir: pathlib.Path, attacks: Sequence[str], usecols=None) -> pd.DataFrame:
    return pd.concat([pd.read_csv(forward_dir / "per_attack" / a / "rows.csv.gz", dtype=DT, usecols=usecols)
                      for a in attacks], ignore_index=True)


def check_forward_frame(df: pd.DataFrame) -> None:
    """Structural invariants of the paired table (fail loudly)."""
    for side in ("control", "attack"):
        hc = [c for c in df.columns if c.startswith(f"{side}_head_")]
        if hc and hc != head_cols(side):
            raise RuntimeError(f"expected exactly the {N_HEADS} {side} head columns of {LAYERS} in order, got {len(hc)}")
    if not set(df.qrel_grade.astype(int)) <= {0, 2, 3}:
        raise RuntimeError("qrel 1 / unjudged rows present")
    grp = np.where(df.qrel_grade.astype(int).isin([2, 3]), RELEVANT, NONRELEVANT)
    if not (grp == df.relevance_group.values).all():
        raise RuntimeError("relevance_group does not match qrel_grade")
    if not np.allclose(df.delta_score, df.score_attack - df.score_control, rtol=0, atol=1e-9):
        raise RuntimeError("delta_score != score_attack - score_control")
    if not (df.successful.astype(bool).values == (df.delta_score.values > 0)).all():
        raise RuntimeError("successful != (delta_score > 0)")
    if df[["attack_name", "pair_id"]].duplicated().any():
        raise RuntimeError("duplicate (attack, qid, docid) instance")


# ---- reference fit + z-scores -----------------------------------------------------------------

def reference_mask(df: pd.DataFrame) -> np.ndarray:
    """Rows whose CONTROL enters the genuine-relevance reference: qrel 2/3 AND successful attack."""
    return (df.relevance_group.values == RELEVANT) & (df.delta_score.values > 0)


def outer_zscores(X: np.ndarray, fold: np.ndarray, X_ref: np.ndarray, fold_ref: np.ndarray, n_folds: int = N_FOLDS):
    """z of every row of X against the reference fitted WITHOUT the row's own fold. Returns (z, fits)."""
    z = np.full(X.shape, np.nan)
    fits = {}
    for f in range(n_folds):
        mu, sd, ok = fit_reference(X_ref[fold_ref != f])
        fits[f] = (mu, sd, ok)
        z[fold == f] = zscores(X[fold == f], mu, sd, ok)
    return z, fits


def states(z: np.ndarray, k: float = K_ABNORMAL) -> np.ndarray:
    """Per head: 0 normal, -1 low abnormal (z < -k), +1 high abnormal (z > k). NaN heads = normal."""
    zz = np.nan_to_num(z, nan=0.0)
    return np.where(zz > k, 1, np.where(zz < -k, -1, 0)).astype(np.int8)


def transition_counts(s_ctl: np.ndarray, s_atk: np.ndarray) -> Dict[str, np.ndarray]:
    """Per-row counts of the four head-state transitions control -> attack (+ low/high split)."""
    a0, a1 = s_ctl != 0, s_atk != 0
    na = ~a0 & a1
    an = a0 & ~a1
    return {"n_normal_to_normal": (~a0 & ~a1).sum(1), "n_normal_to_abnormal": na.sum(1),
            "n_abnormal_to_normal": an.sum(1), "n_abnormal_to_abnormal": (a0 & a1).sum(1),
            "n_normal_to_abnormal_low": (na & (s_atk < 0)).sum(1), "n_normal_to_abnormal_high": (na & (s_atk > 0)).sum(1),
            "n_abnormal_low_to_normal": (an & (s_ctl < 0)).sum(1), "n_abnormal_high_to_normal": (an & (s_ctl > 0)).sum(1)}


# ---- top-k detector (training data only) -------------------------------------------------------

def topk_training_selection(zp_inner: np.ndarray, zn_inner: np.ndarray, k_abn: float, ks: Sequence[int]):
    """
    From INNER held-out z of the training folds only: head order, and for every k the threshold
    T_k (flag if top-k abnormal count >= T_k; exp16lib.anomaly.choose_threshold, balanced accuracy)
    and the inner AUROC_k (used only to pick a single k*).
    """
    order = head_ranking_from_z(zp_inner, zn_inner, k_abn)
    T, inner_auc = {}, {}
    for k in ks:
        sel = order[:k]
        cp = abnormal_counts(zp_inner[:, sel], k_abn)["abnormal_head_count"]
        cn = abnormal_counts(zn_inner[:, sel], k_abn)["abnormal_head_count"]
        T[k] = choose_threshold(cp, cn, k)
        inner_auc[k] = auroc(cp, cn)
    return order, T, inner_auc


def f1_from(tp: int, fp: int, fn: int) -> Dict[str, float]:
    p = tp / (tp + fp) if tp + fp else np.nan
    r = tp / (tp + fn) if tp + fn else np.nan
    f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else np.nan
    return {"precision": p, "recall": r, "f1": f1}


# ---- descriptive aggregation --------------------------------------------------------------------

def cluster_se(x: np.ndarray, clusters: np.ndarray) -> float:
    """Cluster-robust SE of the mean of x (clusters = queries): sqrt(sum_c (sum_{i in c} (x_i - xbar))^2) / n."""
    x = np.asarray(x, float)
    n = len(x)
    if n < 2:
        return np.nan
    r = pd.Series(x - x.mean()).groupby(np.asarray(clusters)).sum().values
    G = len(r)
    if G < 2:
        return np.nan
    return float(np.sqrt(G / (G - 1) * (r ** 2).sum()) / n)


def doc_level(inst: pd.DataFrame) -> pd.DataFrame:
    """One row per (qid, docid): aggregates of its SUCCESSFUL attack instances (equal-document weight downstream)."""
    g = inst.groupby(["relevance_group", "qid", "docid", "pair_id", "qrel_grade"])
    d = g.agg(n_successful_attacks=("attack_name", "size"),
              mean_control_abnormal_count=("control_abnormal_count", "mean"),
              mean_attack_abnormal_count=("attack_abnormal_count", "mean"),
              mean_paired_change=("abnormal_count_change", "mean"),
              median_paired_change=("abnormal_count_change", "median"),
              mean_delta_score=("delta_score", "mean")).reset_index()
    return d


def dist_summary(x) -> Dict[str, float]:
    x = np.asarray(x, float)
    q = np.percentile(x, [25, 50, 75])
    return {"n": int(len(x)), "mean": float(x.mean()), "sd": float(x.std(ddof=1)) if len(x) > 1 else np.nan,
            "q1": float(q[0]), "median": float(q[1]), "q3": float(q[2]), "min": float(x.min()), "max": float(x.max())}


# ---- old-output protection ---------------------------------------------------------------------

def _sha(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fingerprint_old_outputs(out_dir: pathlib.Path) -> Dict[str, str]:
    """sha256 of every pre-existing Exp 16 output file (the paired directories are excluded)."""
    skip = [out_dir / p for p in PAIRED_OUTPUTS]
    res = {}
    for p in sorted(out_dir.rglob("*")):
        if p.is_file() and not any(p == s or s in p.parents for s in skip):
            res[str(p.relative_to(out_dir))] = _sha(p)
    return res
