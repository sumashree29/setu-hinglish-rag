"""
Retrieval quality metrics — Recall@k, MRR, nDCG@k.
OWNER: R1 seeds this in Phase 1 (§3.6), R3 extends/owns it from Phase 3 onward.
"""
from typing import List


def recall_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    """TODO: fraction of `relevant` docs present in retrieved[:k].
    Alt: use ranx's built-in evaluation harness instead of hand-rolling
    (plan §3.6 alt #3) if you want BEIR-convention compatibility."""
    raise NotImplementedError("R1: implement Recall@k")


def mrr(retrieved: List[str], relevant: List[str]) -> float:
    """TODO: 1 / rank of first relevant doc found in `retrieved` (0 if none)."""
    raise NotImplementedError("R1: implement MRR")


def ndcg_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    """TODO: standard nDCG@k with binary relevance (or graded, if you have it)."""
    raise NotImplementedError("R1: implement nDCG@k")


def confidence_proxy(scores: List[float], method: str = "margin") -> float:
    """
    Retrieval-confidence signal used by the LAG classifier and SETU controller.
    method="margin": score[0] - score[1] (rank-1 vs rank-2 gap)
    method="entropy": entropy over top-k normalized scores
    TODO (R3, Phase 3 §5.2): implement both — the plan explicitly wants both
    reported side by side, not just one picked upfront.
    """
    raise NotImplementedError("R3: implement confidence proxy (margin + entropy)")
