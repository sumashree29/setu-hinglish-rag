"""
CARF — CMI-Aware Rank Fusion.
OWNER: R2 | PHASE: 3 (plan §5.1)

Baseline: unweighted Reciprocal Rank Fusion (RRF).
CARF v1: hand-specified CMI/LID-entropy-conditioned weights.
CARF v2: weights fit via ridge/logistic regression against best pilot-set weighting.
"""
from typing import List, Dict


def rrf_baseline(rankings: List[List[str]], k: int = 60) -> List[str]:
    """
    Unweighted Reciprocal Rank Fusion across multiple ranked lists.
    Score of a document: sum(1 / (k + rank_in_list_i)) across lists; sort desc.
    """
    scores = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking):
            scores[doc] = scores.get(doc, 0.0) + 1.0 / (k + rank)
    sorted_docs = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    return sorted_docs


def carf_v1(
    raw_ranking: List[str],
    corrected_ranking: List[str],
    cmi_score: float,
    cmi_max: float = 50.0
) -> List[str]:
    """
    Hand-specified CMI-conditioned weighting on top of RRF.
    As CMI rises, the corrected ranking is weighted more heavily.
    """
    alpha = min(1.0, max(0.0, cmi_score / cmi_max))
    if alpha == 0.0:
        return list(raw_ranking)
    if alpha == 1.0:
        return list(corrected_ranking)

    k = 60
    scores = {}
    for rank, doc in enumerate(raw_ranking):
        scores[doc] = scores.get(doc, 0.0) + (1.0 - alpha) / (k + rank)
    for rank, doc in enumerate(corrected_ranking):
        scores[doc] = scores.get(doc, 0.0) + alpha / (k + rank)

    sorted_docs = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    return sorted_docs


def fit_carf_v2_weights(pilot_results: List[Dict[str, float]]) -> Dict[str, float]:
    """
    Fit weights via ridge regression against the empirically best pilot-set weighting.
    Each item in pilot_results is expected to contain keys:
      - 'cmi_score': float
      - 'lid_entropy': float
      - 'optimal_alpha': float (target blend weight in [0, 1])
    """
    from sklearn.linear_model import Ridge
    import numpy as np

    if not pilot_results:
        return {"w_cmi": 1.0 / 50.0, "w_entropy": 0.0, "intercept": 0.0}

    X = []
    y = []
    for item in pilot_results:
        X.append([item.get("cmi_score", 0.0), item.get("lid_entropy", 0.0)])
        y.append(item.get("optimal_alpha", 0.5))

    X = np.array(X)
    y = np.array(y)

    model = Ridge(alpha=1.0)
    model.fit(X, y)

    return {
        "w_cmi": float(model.coef_[0]),
        "w_entropy": float(model.coef_[1]),
        "intercept": float(model.intercept_),
    }


def carf_v2(
    rankings: List[List[str]],
    cmi_score: float,
    lid_entropy_score: float,
    weights: Dict[str, float]
) -> List[str]:
    """
    Apply learned CARF v2 weights to blend rankings.
    Assumes rankings is [raw_ranking, corrected_ranking].
    """
    if len(rankings) != 2:
        return rrf_baseline(rankings)

    raw_ranking, corrected_ranking = rankings[0], rankings[1]
    w_cmi = weights.get("w_cmi", 1.0 / 50.0)
    w_entropy = weights.get("w_entropy", 0.0)
    intercept = weights.get("intercept", 0.0)

    alpha = intercept + w_cmi * cmi_score + w_entropy * lid_entropy_score
    alpha = min(1.0, max(0.0, alpha))

    return carf_v1(raw_ranking, corrected_ranking, cmi_score=alpha * 50.0, cmi_max=50.0)
