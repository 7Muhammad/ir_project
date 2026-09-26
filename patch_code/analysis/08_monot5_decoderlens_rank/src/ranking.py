"""
src/ranking.py
==============
Compute the rank of a target passage within its query's candidate set, for a
given encoder layer.

The candidate set is fixed: all candidates keep their CLEAN passage scores.
Only the target passage's score is swapped for the variant being evaluated
(original / padded_control / attack).  The rank is ALWAYS computed against the
whole candidate set — never just the attacked/control pair.

Rank convention
---------------
Rank 1 is best (highest score).  Ties are broken so that the target does not
get an artificially good rank: the target's rank is 1 + (number of other
candidates with a STRICTLY higher score), then ties are resolved by counting
candidates with equal score as ranked above the target only if their docid
sorts before the target's.  In practice exact score ties are essentially
impossible with float scores, so this only matters for the degenerate case.
"""

from __future__ import annotations

from typing import Dict, List


def compute_rank(
    candidate_scores: Dict[str, float],
    target_docid: str,
    target_score: float,
) -> int:
    """
    Rank of the target among the candidate set using ``target_score`` for the
    target and the clean score for every other candidate.

    Parameters
    ----------
    candidate_scores : {docid: clean_score} for the full candidate set
        (includes the target's own clean score, which is overridden).
    target_docid : the docid being ranked.
    target_score : the score to use for the target (variant score).

    Returns
    -------
    int rank, 1 = best.
    """
    higher = 0
    for docid, clean_score in candidate_scores.items():
        if docid == target_docid:
            continue
        if clean_score > target_score:
            higher += 1
        elif clean_score == target_score and docid < target_docid:
            # Deterministic tie-break: equal-score candidate with a smaller
            # docid is ranked above the target.
            higher += 1
    return higher + 1


def rank_target_over_layers(
    candidate_scores_by_layer: List[Dict[str, float]],
    target_docid: str,
    target_scores_by_layer: List[float],
) -> List[int]:
    """
    Compute the target's rank at every encoder layer.

    Parameters
    ----------
    candidate_scores_by_layer : list (len = n_layers) of {docid: clean_score}.
    target_docid : the docid being ranked.
    target_scores_by_layer : list (len = n_layers) of the target's variant score.

    Returns
    -------
    list of int ranks, one per layer.
    """
    ranks: List[int] = []
    for layer_idx, cand_scores in enumerate(candidate_scores_by_layer):
        ranks.append(
            compute_rank(cand_scores, target_docid, target_scores_by_layer[layer_idx])
        )
    return ranks
