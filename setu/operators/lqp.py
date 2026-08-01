"""
LQP — Linear Query-space Projection.
OWNER: R1 | PHASE: 2 (plan §4.1)

Fits W minimizing ||X·W - Y||^2 + lambda*||W||^2 on Hinglish->English parallel
embedding pairs (HinGE, PHINC), then builds a CMI-conditional blend so light
code-mixing barely gets projected and heavy code-mixing gets projected more.
"""
import numpy as np
from sklearn.linear_model import Ridge


def load_parallel_pairs():
    """
    Load HinGE + PHINC parallel query pairs, embed both sides with the same
    embedding model used elsewhere in the pipeline.
    Returns: (X, Y) — Hinglish embeddings, English-equivalent embeddings, same shape.

    TODO (R1): download links in plan §4.1 / §10. Augment with Aksharantar
    transliteration pairs if HinGE+PHINC (~15k pairs) underperforms (alt #3).
    """
    raise NotImplementedError("R1: load + embed HinGE/PHINC pairs")


def fit_lqp(X: np.ndarray, Y: np.ndarray, alpha: float = 1.0) -> Ridge:
    """
    Fit the ridge regression projection matrix W.
    TODO (R1): sklearn.linear_model.Ridge(alpha=alpha).fit(X, Y)
    Alt: RidgeCV for automatic alpha selection instead of hand-tuning (plan alt #2).
    """
    raise NotImplementedError("R1: fit ridge regression")


def apply_lqp(query_embedding: np.ndarray, cmi_score: float, model: Ridge, cmi_max: float = 1.0) -> np.ndarray:
    """
    Apply the CMI-conditional projection:
        alpha = min(1, cmi_score / cmi_max)
        W_eff = (1 - alpha) * I + alpha * W
        return query_embedding @ W_eff

    TODO (R1): implement the blend exactly as in plan §4.1 — this conditional
    blending (not just raw W) is the actual novelty claim, don't skip it.
    """
    raise NotImplementedError("R1: implement CMI-conditional projection")
