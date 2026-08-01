"""
LID-entropy(q) — Shannon entropy over per-token language-ID predictions.
OWNER: R1 | PHASE: 1 (plan §3.5)

High entropy = the model is unsure which language each token belongs to /
tokens are evenly split between languages = more "confusing" mixing than CMI
alone captures.
"""
from typing import List


def load_lid_model():
    """
    Load the IndicLID model once (fasttext + small BERT classifier per the plan,
    §3.3). Cache it — don't reload per-query, it's slow.

    TODO (R1):
        - pip/clone per https://github.com/AI4Bharat/IndicLID
        - Fallback per plan: fasttext lid.176.bin if IndicLID's romanized-script
          model is fussy to set up
    Returns: whatever object token_lid_tags() below needs.
    """
    raise NotImplementedError("R1: load IndicLID here")


def token_lid_tags(query: str, model=None) -> List[str]:
    """Returns one language tag per token, e.g. ['hi','hi','en','hi']."""
    raise NotImplementedError("R1: run IndicLID inference per token")


def lid_entropy(query: str, model=None) -> float:
    """
    Args:
        query: raw query string
        model: object returned by load_lid_model() — pass this in, don't reload here

    Returns:
        Shannon entropy (bits) over the language-tag distribution for this query.

    TODO (R1):
        1. tags = token_lid_tags(query, model)
        2. compute proportion of each unique tag
        3. entropy = -sum(p * log2(p) for p in proportions)
        Smoothing note (plan §3.5 alt #2): if entropy looks noisy on short
        queries, apply add-1 (Laplace) smoothing to the tag counts before
        computing proportions.
    """
    raise NotImplementedError("R1: implement per plan §3.5")
