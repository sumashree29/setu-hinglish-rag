"""
CMI(q) — Code-Mixing Index for a single query.
OWNER: R1 | PHASE: 1 (plan §3.5)

Standard formula (Gambäck & Das): fraction of tokens in the query that come
from the non-matrix (secondary) language. Uses per-token language tags from
token_lid_tags() in lid_entropy.py (shared tagging call, tagged once).
"""
from typing import List

from .lid_entropy import token_lid_tags

def cmi(query: str, model=None) -> float:
    """
    Args:
        query: raw Hinglish query string, e.g. "mera loan status kaise pata karu"
        model: optional LID model object, passed through to token_lid_tags()
    Returns:
        float in [0, 1] -- 0 = monolingual matrix language, higher = more code-mixed.
    """
    tags = token_lid_tags(query, model)
    n = len(tags)

    if n <= 1:
        return 0.0

    from collections import Counter
    counts = Counter(tags)
    matrix_lang, matrix_count = counts.most_common(1)[0]

    return round((n - matrix_count) / n, 4)


def cmi_batch(queries: List[str], model=None) -> List[float]:
    """Convenience wrapper -- same as cmi() but for a list."""
    return [cmi(q, model) for q in queries]


if __name__ == "__main__":
    test_cases = [
        ("What is the interest rate for a savings account", "pure English"),
        ("mera SBI account band karna hai kaise", "mostly Hindi, few English/other"),
        ("Mujhe 50000 rupaye ka loan chahiye SBI se", "balanced mix"),
        ("hai", "single token"),
        ("", "empty"),
    ]
    for q, desc in test_cases:
        print(f"{desc:35s} CMI={cmi(q)}   query={q!r}")