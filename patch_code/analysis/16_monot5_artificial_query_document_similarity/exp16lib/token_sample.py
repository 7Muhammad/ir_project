"""
exp16lib/token_sample.py
=========================
Per-token head outputs for a small, fixed exploration sample (stage 22), so new
query/document metrics can be tried without another forward pass. Promising metrics are
later computed on the fly over the full paired population; pooled vectors are never
bulk-stored.

Stored per sequence (clean input, padded control, attacked input): for EVERY token, the
pre-o_proj output of all 144 encoder heads (EncoderHeadCapture(keep_inputs=True) = the
exact tensor the paired pipeline pools), shape [n_tokens, 12 layers, 12 heads, 64],
fp16. Batch padding is never stored; masked insertion slots of the control ARE stored
(masks tell them apart).

On disk (22_token_sample/):
  heads.npy        [total_tokens, 12, 12, 64] float16, memory-mappable
  tokens.npz       input_ids, attention_mask, query_mask, doc_mask, inserted_mask  (all [total_tokens])
  sequences.csv    one row per sequence: seq_id, token_offset, n_tokens, kind (clean/control/attack),
                   attack_name, pair_id, qid, docid, qrel_grade, relevance_group, score, delta_score, successful, ...

Usage:
  from exp16lib.token_sample import TokenSample
  ts = TokenSample(".../outputs/22_token_sample")
  seq = ts[0]                                   # dict: heads [T,12,12,64] (memmap), masks, meta
  q = seq["heads"][seq["query_mask"], 11, 2]    # query tokens, layer 11, head 2 -> [n_q, 64]
  for seq in ts.iter(kind="control", relevance_group="relevant"): ...
"""

from __future__ import annotations

import pathlib
import random
from typing import Dict, Iterator, List

import numpy as np
import pandas as pd

N_LAYERS, N_HEADS_PER_LAYER, D_KV = 12, 12, 64

# Balanced 12-attack subset: 3 positions x reps {1, 5} x 2 tokens; each of the 7 tokens appears
# once or twice (relevant, true, information, false, bar: 2; important, relevance: 1).
DEFAULT_ATTACKS = ["relevant_start_1", "true_start_1", "information_start_5", "false_start_5",
                   "bar_end_1", "important_end_1", "relevance_end_5", "relevant_end_5",
                   "information_random_1", "true_random_1", "false_random_5", "bar_random_5"]


def sample_pairs(base: List[Dict], per_group: int, seed: int) -> List[Dict]:
    """Up to `per_group` qrel 2/3 and qrel 0 base pairs per query; random.Random(f'{seed}:{qid}') over sorted docids."""
    by = {}
    for r in base:
        by.setdefault((r["qid"], r["relevance_group"]), []).append(r)
    out = []
    for qid in sorted({r["qid"] for r in base}, key=lambda x: (len(x), x)):
        rng = random.Random(f"{seed}:{qid}")
        for g in ("relevant", "nonrelevant"):
            rs = sorted(by.get((qid, g), []), key=lambda r: r["docid"])
            out += sorted(rng.sample(rs, min(per_group, len(rs))), key=lambda r: r["docid"])
    return out


class TokenSample:
    """Read-only access to a stage-22 sample (heads.npy is memory-mapped, never fully loaded)."""

    def __init__(self, path):
        self.dir = pathlib.Path(path)
        self.meta = pd.read_csv(self.dir / "sequences.csv", dtype={"qid": str, "docid": str, "pair_id": str,
                                                                   "attack_name": str, "attack_token": str,
                                                                   "attack_position": str}, low_memory=False)
        if "successful" in self.meta:                              # empty for clean rows -> nullable boolean
            self.meta["successful"] = self.meta.successful.map({True: True, False: False, "True": True,
                                                                "False": False}).astype("boolean")
        self.heads = np.load(self.dir / "heads.npy", mmap_mode="r")
        t = np.load(self.dir / "tokens.npz")
        self.tok = {k: t[k] for k in t.files}

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, i: int) -> Dict:
        m = self.meta.iloc[i]
        a, b = int(m.token_offset), int(m.token_offset + m.n_tokens)
        d = {"heads": self.heads[a:b], **{k: (v[a:b].astype(bool) if k.endswith("mask") else v[a:b]) for k, v in self.tok.items()}}
        d["meta"] = m.to_dict()
        return d

    def select(self, **filters) -> pd.DataFrame:
        m = self.meta
        for k, v in filters.items():
            m = m[m[k].isin(v if isinstance(v, (list, tuple, set)) else [v])]
        return m

    def iter(self, **filters) -> Iterator[Dict]:
        for i in self.select(**filters).index:
            yield self[i]

    @staticmethod
    def pooled_cosine(seq: Dict, layer: int, head: int) -> float:
        """The paired pipeline's metric recomputed from stored tokens (fp32 means, float64 cosine)."""
        h = seq["heads"][:, layer, head].astype(np.float32)
        q, d = h[seq["query_mask"]].mean(0).astype(np.float64), h[seq["doc_mask"]].mean(0).astype(np.float64)
        return float(q @ d / max(np.linalg.norm(q) * np.linalg.norm(d), 1e-12))
