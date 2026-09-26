"""
exp12lib/stopwords.py
======================
Fixed, documented English stopword list used throughout Experiment 12 for the
content-vs-stopword classification of query words (see PART "QUERY WORD
CLASSIFICATION" in the experiment prompt).

Why a hardcoded list instead of nltk/sklearn
----------------------------------------------
Neither nltk nor scikit-learn is installed in the project's conda environment
(advseq2seq — verified: `python -c "import nltk"` -> ModuleNotFoundError).
Adding a new runtime dependency (plus, for nltk, a corpus download step) for a
single fixed word list is not worth the added fragility, especially for a
resume-safe SLURM pipeline that must not depend on network access at run time.

Provenance
----------
This is verbatim NLTK's `nltk.corpus.stopwords.words("english")` list (179
words, NLTK data package "stopwords", checked against the standard
distribution). Using a well-known, citable, unchanging list satisfies the
experiment's requirement for "one fixed, documented English stopword
list/library" without adding a dependency. If nltk is later added to the
environment, `nltk.corpus.stopwords.words("english")` should reproduce this
exact set (order-independent — comparisons here always use the frozenset).
"""

from __future__ import annotations

NLTK_ENGLISH_STOPWORDS = [
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you",
    "you're", "you've", "you'll", "you'd", "your", "yours", "yourself",
    "yourselves", "he", "him", "his", "himself", "she", "she's", "her",
    "hers", "herself", "it", "it's", "its", "itself", "they", "them",
    "their", "theirs", "themselves", "what", "which", "who", "whom",
    "this", "that", "that'll", "these", "those", "am", "is", "are", "was",
    "were", "be", "been", "being", "have", "has", "had", "having", "do",
    "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or",
    "because", "as", "until", "while", "of", "at", "by", "for", "with",
    "about", "against", "between", "into", "through", "during", "before",
    "after", "above", "below", "to", "from", "up", "down", "in", "out",
    "on", "off", "over", "under", "again", "further", "then", "once",
    "here", "there", "when", "where", "why", "how", "all", "any", "both",
    "each", "few", "more", "most", "other", "some", "such", "no", "nor",
    "not", "only", "own", "same", "so", "than", "too", "very", "s", "t",
    "can", "will", "just", "don", "don't", "should", "should've", "now",
    "d", "ll", "m", "o", "re", "ve", "y", "ain", "aren", "aren't",
    "couldn", "couldn't", "didn", "didn't", "doesn", "doesn't", "hadn",
    "hadn't", "hasn", "hasn't", "haven", "haven't", "isn", "isn't", "ma",
    "mightn", "mightn't", "mustn", "mustn't", "needn", "needn't",
    "shan", "shan't", "shouldn", "shouldn't", "wasn", "wasn't", "weren",
    "weren't", "won", "won't", "wouldn", "wouldn't",
]

STOPWORDS = frozenset(w.lower() for w in NLTK_ENGLISH_STOPWORDS)


def is_stopword(normalized_word: str) -> bool:
    """`normalized_word` must already be lower-cased (see query_words.normalize_word)."""
    return normalized_word in STOPWORDS
