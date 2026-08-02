"""
Retrieval quality metrics — Recall@k, MRR, nDCG@k.
OWNER: R1 seeds this in Phase 1 (§3.6), R3 extends/owns it from Phase 3 onward.
"""
import math
from typing import List


def recall_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    """Fraction of `relevant` docs present in retrieved[:k].
    Alt: use ranx's built-in evaluation harness instead of hand-rolling
    (plan §3.6 alt #3) if you want BEIR-convention compatibility."""
    raise NotImplementedError("R1: implement Recall@k")


def mrr(retrieved: List[str], relevant: List[str]) -> float:
    """1 / rank of first relevant doc found in `retrieved` (0 if none)."""
    raise NotImplementedError("R1: implement MRR")


def ndcg_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    """Standard nDCG@k with binary relevance (or graded, if you have it)."""
    raise NotImplementedError("R1: implement nDCG@k")


def confidence_proxy(scores: List[float], method: str = "margin") -> float:
    """
    Retrieval-confidence signal used by the LAG classifier and SETU controller.
    method="margin": score[0] - score[1] (rank-1 vs rank-2 gap)
    method="entropy": entropy over top-k normalized scores
    """
    if not scores:
        return 0.0
        
    sorted_scores = sorted(scores, reverse=True)
    
    if method == "margin":
        if len(sorted_scores) < 2:
            return float(sorted_scores[0]) if sorted_scores else 0.0
        return float(sorted_scores[0] - sorted_scores[1])
        
    elif method == "entropy":
        # Normalize scores to sum to 1
        total = sum(sorted_scores)
        if total <= 0:
            exp_scores = [math.exp(s) for s in sorted_scores]
            total_exp = sum(exp_scores)
            probs = [e / total_exp for e in exp_scores]
        else:
            probs = [s / total for s in sorted_scores]
            
        entropy_val = 0.0
        for p in probs:
            if p > 0:
                entropy_val -= p * math.log2(p)
        return float(entropy_val)
        
    else:
        raise ValueError(f"Unknown method: {method}")
