"""
Retrieval quality metrics & confidence signals.
OWNER: R1 seeds this in Phase 1 (§3.6), R3 extends/owns it from Phase 3 onward.

NOTE: Retrieval evaluation metrics (Recall@k, MRR, nDCG@k) are computed via
ranx.evaluate() across all evaluation scripts (e.g., compare_setu_v1_v2.py,
evaluate_operators_standalone.py) for BEIR-convention compatibility.
"""
import math
from typing import List


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
        if min(sorted_scores) < 0 or sum(sorted_scores) <= 0:
            exp_scores = [math.exp(s) for s in sorted_scores]
            total_exp = sum(exp_scores)
            probs = [e / total_exp for e in exp_scores]
        else:
            total = sum(sorted_scores)
            probs = [s / total for s in sorted_scores]
            
        entropy_val = 0.0
        for p in probs:
            if p > 0:
                entropy_val -= p * math.log2(p)
        return float(entropy_val)
        
    else:
        raise ValueError(f"Unknown method: {method}")
