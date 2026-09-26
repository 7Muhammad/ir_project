"""
exp16lib/inputs.py
===================
Token ids, attention masks and the query / document POOLING masks for every
input type, built only with existing project machinery:

  prompt text     src.model_utils.build_monot5_input        (Exp 01)
  control/attack  src.model_utils.build_padded_control_and_attack_encodings_general
                  (Exp 01 token-level alignment; the same construction that
                  produced the cached Exp 01 control/attack scores)
  spans           exp6lib.spans.find_query_and_doc_spans    (Exp 06; exact
                  token-prefix probes, never string search on the output)

Pooling masks (DECISIONS 4-7):
  query mask      = range(*query_span): the query TEXT only. "▁ Query :",
                    "▁Document :", "▁Relevan t :" and </s> are excluded.
  clean/qrel doc  = range(*doc_span_original): the passage text only.
  attack doc      = range(*doc_span_control_attack): passage tokens AND every
                    injected attack token (all active).
  control doc     = range(*doc_span_control_attack) AND attention_mask == 1:
                    the masked insertion slots never contribute.

Prompts longer than max_length raise EncodingError (never truncated
silently; truncation would cut the document and the "Relevant:" suffix).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import torch

import exp16lib  # noqa: F401
from exp6lib.spans import find_query_and_doc_spans
from src.model_utils import build_monot5_input, build_padded_control_and_attack_encodings_general

TAIL_LEN = 4  # "▁Relevan", "t", ":", "</s>" (Exp 06 template_positions ground truth)


class EncodingError(RuntimeError):
    pass


@dataclass
class EncodedSeq:
    input_ids: List[int]
    attention_mask: List[int]
    query_mask: List[int]
    doc_mask: List[int]
    meta: Dict = field(default_factory=dict)

    @property
    def seq_len(self) -> int:
        return len(self.input_ids)

    @property
    def n_query_tokens(self) -> int:
        return int(sum(self.query_mask))

    @property
    def n_doc_tokens(self) -> int:
        return int(sum(self.doc_mask))


def _range_mask(n: int, lo: int, hi: int) -> List[int]:
    return [1 if lo <= i < hi else 0 for i in range(n)]


def encode_clean(tokenizer, query: str, passage: str, max_length: int) -> EncodedSeq:
    """Type-A prompt (also used for qrel-judged documents)."""
    ids = tokenizer.encode(build_monot5_input(query, passage), add_special_tokens=True)
    if len(ids) > max_length:
        raise EncodingError(f"clean prompt has {len(ids)} tokens > max_length={max_length}")
    spans = find_query_and_doc_spans(tokenizer, query, passage, 0)
    if spans.status != "ok":
        raise EncodingError(f"span detection failed: {spans.reason}")
    q0, q1 = spans.query_span
    d0, d1 = spans.doc_span_original
    n = len(ids)
    if d1 != n - TAIL_LEN:
        raise EncodingError(f"unexpected tail: doc_end={d1}, seq_len={n}")
    return EncodedSeq(ids, [1] * n, _range_mask(n, q0, q1), _range_mask(n, d0, d1),
                      {"query_span": [q0, q1], "doc_span": [d0, d1]})


def encode_attack_and_control(tokenizer, query: str, passage: str, attacked_passage: str,
                              max_length: int):
    """
    Returns (attack_seq, control_seq, info). Invariants (fail loudly):
      * Exp 01 alignment succeeds; no truncation;
      * removing the inserted positions A from the attacked prompt gives the clean prompt;
      * control == attack outside A, control has pad + mask 0 exactly on A;
      * the attack doc span holds exactly n_passage + n_inserted tokens;
      * A lies inside the document field (or starts at the "Document:" colon —
        the benign Exp 15 "boundary shift", flagged, see DECISIONS 40).
    """
    cpu = torch.device("cpu")
    c_enc, a_enc, align = build_padded_control_and_attack_encodings_general(
        tokenizer=tokenizer, query=query, passage=passage, attacked_passage=attacked_passage,
        max_length=10 ** 9, device=cpu)
    if align.status != "ok":
        raise EncodingError(f"attack alignment failed: {align.reason}")
    a_ids = a_enc["input_ids"][0].tolist()
    c_ids = c_enc["input_ids"][0].tolist()
    c_mask = c_enc["attention_mask"][0].tolist()
    n = len(a_ids)
    if n > max_length:
        raise EncodingError(f"attacked prompt has {n} tokens > max_length={max_length}")
    spans = find_query_and_doc_spans(tokenizer, query, passage, align.n_inserted)
    if spans.status != "ok":
        raise EncodingError(f"span detection failed: {spans.reason}")
    q0, q1 = spans.query_span
    d0, d1 = spans.doc_span_control_attack
    n_passage = spans.doc_span_original[1] - spans.doc_span_original[0]
    A = sorted(int(p) for p in align.inserted_positions)
    Aset = set(A)

    original_ids = tokenizer.encode(build_monot5_input(query, passage), add_special_tokens=True)
    if [t for i, t in enumerate(a_ids) if i not in Aset] != original_ids:
        raise EncodingError("removing inserted positions from the attacked prompt does not give the clean prompt")
    if len(c_ids) != n or d1 != n - TAIL_LEN or d1 - d0 != n_passage + len(A):
        raise EncodingError(f"layout mismatch: n={n}, doc_span=({d0},{d1}), n_passage={n_passage}, n_ins={len(A)}")
    for i in range(n):
        if i in Aset:
            if c_ids[i] != tokenizer.pad_token_id or c_mask[i] != 0:
                raise EncodingError(f"control not padded/masked at inserted position {i}")
        elif c_ids[i] != a_ids[i] or c_mask[i] != 1:
            raise EncodingError(f"control differs from attack at non-inserted position {i}")
    boundary_shift = min(A) < d0
    if not (min(A) >= q1 and max(A) < d1) or (boundary_shift and min(A) != d0 - 1):
        raise EncodingError(f"inserted positions {A[:5]}... outside the document field ({d0},{d1})")

    q_mask = _range_mask(n, q0, q1)
    a_doc = _range_mask(n, d0, d1)
    c_doc = [m * am for m, am in zip(a_doc, c_mask)]
    info = {
        "seq_len": n, "n_inserted": len(A), "inserted_positions": A,
        "query_span": [q0, q1], "doc_span": [d0, d1], "n_passage_tokens": n_passage,
        "alignment_boundary_shift": bool(boundary_shift),
    }
    attack = EncodedSeq(a_ids, [1] * n, q_mask, a_doc, info)
    control = EncodedSeq(c_ids, c_mask, list(q_mask), c_doc, info)
    return attack, control, info


def collate(seqs: List[EncodedSeq], pad_id: int, device: torch.device) -> Dict[str, torch.Tensor]:
    """Right-pad a batch. Batch padding gets attention 0 and is outside both pooling masks."""
    S = max(s.seq_len for s in seqs)

    def pad(xs, v):
        return torch.tensor([x + [v] * (S - len(x)) for x in xs], dtype=torch.long, device=device)

    return {
        "input_ids": pad([s.input_ids for s in seqs], pad_id),
        "attention_mask": pad([s.attention_mask for s in seqs], 0),
        "query_mask": pad([s.query_mask for s in seqs], 0),
        "doc_mask": pad([s.doc_mask for s in seqs], 0),
    }
