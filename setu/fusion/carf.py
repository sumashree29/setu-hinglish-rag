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
    TODO (R2): score(doc) = sum(1 / (k + rank_in_list_i)) across lists;
    sort desc. ~10 lines from scratch (plan alt #1) or use ranx's Run/Qrels
    objects if you'd rather not hand-roll it.
    """
    raise NotImplementedError("R2: implement RRF")


def carf_v1(rankings: List[List[str]], cmi_score: float, lid_entropy_score: float) -> List[str]:
    """
    Hand-specified CMI/LID-entropy-conditioned weighting on top of RRF.
    TODO (R2): design a weighting rule — e.g. weight the "corrected" ranking
    list more heavily as CMI rises. Document the exact rule you pick, since
    this is a paper contribution, not just an implementation detail.
    """
    raise NotImplementedError("R2: implement CARF v1")


def fit_carf_v2_weights(pilot_results) -> Dict[str, float]:
    """Fit weights via ridge/logistic regression against the empirically best
    pilot-set weighting, rather than hand-specifying them.
    TODO (R2): implement after CARF v1 exists and pilot data is available."""
    raise NotImplementedError("R2: fit CARF v2 weights")


def carf_v2(rankings: List[List[str]], cmi_score: float, lid_entropy_score: float, weights: Dict[str, float]) -> List[str]:
    raise NotImplementedError("R2: apply learned CARF v2 weights")
