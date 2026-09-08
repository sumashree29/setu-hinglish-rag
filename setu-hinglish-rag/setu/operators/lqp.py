"""
LQP — Linear Query-space Projection.
OWNER: R1 | PHASE: 2 (plan §4.1)

Fits W minimizing ||X·W - Y||^2 + lambda*||W||^2 on Hinglish->English parallel
embedding pairs (PHINC, and HinGE if you can source it), then builds a
CMI-conditional blend: light code-mixing barely gets projected, heavy
code-mixing gets projected more.

    W_eff(q) = (1 - alpha) * I + alpha * W,    alpha = min(1, CMI(q) / CMI_max)
    projected_embedding = query_embedding @ W_eff
"""
from typing import List, Tuple
import numpy as np
from sklearn.linear_model import Ridge, RidgeCV


def load_parallel_pairs_phinc(embed_fn, max_pairs: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load PHINC (Hinglish -> English) parallel sentence pairs and embed both sides.
    Requires: pip install datasets
    Run locally/Colab — needs real internet access to huggingface.co.
    """
    from datasets import load_dataset

    ds = load_dataset("LingoIITGN/PHINC", split="train")
    if max_pairs:
        ds = ds.select(range(min(max_pairs, len(ds))))

    hinglish_col = "Sentence" if "Sentence" in ds.column_names else "Hinglish Code-Mixed Sentence"
    english_col = "English_Translation" if "English_Translation" in ds.column_names else "Human Translated English Sentence"

    hinglish_sents = [row[hinglish_col] for row in ds]
    english_sents = [row[english_col] for row in ds]

    X = embed_fn(hinglish_sents)
    Y = embed_fn(english_sents)
    return np.asarray(X), np.asarray(Y)


def fit_lqp(X: np.ndarray, Y: np.ndarray, alpha_reg: float = 1.0, use_cv: bool = False) -> Ridge:
    """Fit the ridge regression projection matrix W: Y ≈ X @ W"""
    if use_cv:
        model = RidgeCV(alphas=np.logspace(-3, 3, 13))
    else:
        model = Ridge(alpha=alpha_reg)
    model.fit(X, Y)
    return model


def apply_lqp(query_embedding: np.ndarray, cmi_score: float, model: Ridge, cmi_max: float = 1.0) -> np.ndarray:
    """Apply the CMI-conditional projection to one query embedding.
    cmi_score comes from setu.diagnosis.cmi.cmi(), which returns [0,1]
    (fraction of non-majority-language tokens). NOTE: this scale changed
    twice during Phase 1 (originally 0-100, briefly 0-50, now 0-1) --
    always check setu/diagnosis/cmi.py's current docstring if unsure."""
    d = query_embedding.shape[0]
    W = model.coef_.T
    alpha = min(1.0, cmi_score / cmi_max)
    W_eff = (1 - alpha) * np.eye(d) + alpha * W
    return query_embedding @ W_eff


def apply_lqp_batch(query_embeddings: np.ndarray, cmi_scores: List[float], model: Ridge, cmi_max: float = 1.0) -> np.ndarray:
    """Vectorized version for many queries at once."""
    n, d = query_embeddings.shape
    W = model.coef_.T
    I = np.eye(d)
    alphas = np.minimum(1.0, np.asarray(cmi_scores) / cmi_max)
    out = np.empty_like(query_embeddings, dtype=float)
    for i in range(n):
        W_eff = (1 - alphas[i]) * I + alphas[i] * W
        out[i] = query_embeddings[i] @ W_eff
    return out
