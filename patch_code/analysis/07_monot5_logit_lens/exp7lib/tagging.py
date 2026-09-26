"""
exp7lib/tagging.py
====================
Bucket classification of top-k logit-lens tokens: attack / query / document
/ stopword / other. See DECISIONS.md for the priority order (attack checked
first — the attack-grid words are ordinary English words that can
coincidentally overlap with query/document vocabulary) and for why matching
uses a small hardcoded stemmer/stopword list instead of NLTK (not installed
in this project's environment; not worth adding for a fallback-only
heuristic).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import FrozenSet, Set

# A standard, fixed English stopword list (not exhaustive; covers the
# high-frequency function words that would otherwise dominate "other").
STOPWORDS: FrozenSet[str] = frozenset("""
a about above after again against all am an and any are aren't as at be
because been before being below between both but by can't cannot could
couldn't did didn't do does doesn't doing don't down during each few for
from further had hadn't has hasn't have haven't having he he'd he'll he's
her here here's hers herself him himself his how how's i i'd i'll i'm i've
if in into is isn't it it's its itself let's me more most mustn't my
myself no nor not of off on once only or other ought our ours ourselves
out over own same shan't she she'd she'll she's should shouldn't so some
such than that that's the their theirs them themselves then there there's
these they they'd they'll they're they've this those through to too under
until up very was wasn't we we'd we'll we're we've were weren't what what's
when when's where where's which while who who's whom why why's with won't
would wouldn't you you'd you'll you're you've your yours yourself
yourselves
""".split())


def simple_stem(word: str) -> str:
    """
    Crude, dependency-free suffix stripper — not linguistically correct
    stemming, only needs to catch the plural/case/inflection variants the
    task calls out (e.g. "documents" -> "document", "running" -> "runn").
    """
    w = word.lower()
    for suffix in ("ing", "es", "ed", "ly", "s"):
        if w.endswith(suffix) and len(w) - len(suffix) >= 3:
            return w[: -len(suffix)]
    return w


def normalize_sp_token(sp_token: str) -> str:
    """Strip SentencePiece's leading '▁' (space marker) and lowercase."""
    return sp_token.replace("▁", "").lower()


def _words(text: str) -> Set[str]:
    return set(re.findall(r"[a-zA-Z0-9]+", text.lower()))


@dataclass
class TaggingVocab:
    query_exact: Set[str] = field(default_factory=set)
    query_stems: Set[str] = field(default_factory=set)
    doc_exact: Set[str] = field(default_factory=set)
    doc_stems: Set[str] = field(default_factory=set)
    attack_exact: Set[str] = field(default_factory=set)
    attack_stems: Set[str] = field(default_factory=set)


def build_tagging_vocab(query: str, passage: str, attack_token: str) -> TaggingVocab:
    query_words = _words(query)
    doc_words = _words(passage)
    attack_words = {attack_token.lower()}
    return TaggingVocab(
        query_exact=query_words, query_stems={simple_stem(w) for w in query_words},
        doc_exact=doc_words, doc_stems={simple_stem(w) for w in doc_words},
        attack_exact=attack_words, attack_stems={simple_stem(w) for w in attack_words},
    )


def tag_token(sp_token: str, vocab: TaggingVocab) -> str:
    """
    Classify one top-k SentencePiece token as 'attack' | 'query' | 'document'
    | 'stopword' | 'other', priority order per DECISIONS.md.
    """
    word = normalize_sp_token(sp_token)
    if not word:
        return "other"
    stem = simple_stem(word)

    if word in vocab.attack_exact or stem in vocab.attack_stems:
        return "attack"
    if word in vocab.query_exact or stem in vocab.query_stems:
        return "query"
    if word in vocab.doc_exact or stem in vocab.doc_stems:
        return "document"
    if word in STOPWORDS:
        return "stopword"
    return "other"


def bucket_composition(sp_tokens: list, vocab: TaggingVocab) -> dict:
    """Percent of `sp_tokens` in each bucket (sums to 100.0, or 0.0 dict if empty)."""
    buckets = {"attack": 0, "query": 0, "document": 0, "stopword": 0, "other": 0}
    if not sp_tokens:
        return {k: 0.0 for k in buckets}
    for tok in sp_tokens:
        buckets[tag_token(tok, vocab)] += 1
    n = len(sp_tokens)
    return {k: 100.0 * v / n for k, v in buckets.items()}
