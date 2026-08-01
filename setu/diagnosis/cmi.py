"""
CMI(q) — Code-Mixing Index for a single query.
OWNER: R1 | PHASE: 1 (plan §3.5)

Standard formula (Gambäck & Das): fraction of tokens in the query that come
from the non-matrix (secondary) language. Needs per-token language tags —
get those from IndicLID (see setu/diagnosis/lid_entropy.py, which should
share the same tagging call so you don't tag twice).
"""
from typing import List


def cmi(query: str) -> float:
    """
    Args:
        query: raw Hinglish query string, e.g. "mera loan status kaise pata karu"

    Returns:
        float in [0, 1] — 0 = monolingual matrix language, higher = more code-mixed.

    TODO (R1):
        1. Tokenize `query`
        2. Get per-token language tags (reuse the IndicLID call from lid_entropy.py)
        3. matrix_lang = most frequent tag
        4. return (count of tokens NOT matrix_lang) / (total tokens)
        Edge case: if query has 0 or 1 tokens, define CMI as 0.0 (no mixing possible).
    """
    raise NotImplementedError("R1: implement CMI per plan §3.5")


def cmi_batch(queries: List[str]) -> List[float]:
    """Convenience wrapper — same as cmi() but for a list. Don't re-implement
    the logic here, just call cmi() per query (or batch the IndicLID call for speed)."""
    raise NotImplementedError("R1: implement after cmi() is working")
