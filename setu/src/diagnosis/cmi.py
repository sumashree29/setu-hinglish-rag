"""
CMI — Code-Mixing Index (Gambäck & Das, 2014)

CMI(x) = 100 * (1 - max(w_i) / (n - u))   if n > u
CMI(x) = 0                                 if n == u  (fully language-independent, e.g. all numbers/NEs)

Where:
    n      = total number of tokens
    u      = number of language-independent tokens (named entities, numbers, punctuation)
    w_i    = token count for language i
    max(w_i) = count of the most frequent language in the utterance

Expects a token-level language tag list as input (produced by the LID tagger
we'll build next). Kept decoupled from any specific LID model so it can be
unit-tested with hand-crafted tag sequences first.
"""
from collections import Counter
from typing import List

LANG_INDEPENDENT_TAGS = {"NE", "NUM", "PUNCT", "OTHER"}


def compute_cmi(tags: List[str]) -> float:
    """
    Compute CMI given a list of per-token language tags.
    e.g. tags = ["HI", "HI", "EN", "NE", "HI", "PUNCT"]
    Returns CMI in [0, 100]. 0 = monolingual or fully language-independent.
    """
    n = len(tags)
    if n == 0:
        return 0.0

    u = sum(1 for t in tags if t in LANG_INDEPENDENT_TAGS)

    if n == u:
        return 0.0

    lang_tags = [t for t in tags if t not in LANG_INDEPENDENT_TAGS]
    counts = Counter(lang_tags)
    max_wi = max(counts.values())

    cmi = 100.0 * (1 - (max_wi / (n - u)))
    return round(cmi, 2)


def cmi_band(cmi: float, bands) -> str:
    """Assign a CMI score to a named band, using bands from config.CMI_BANDS."""
    for lo, hi, name in bands:
        if lo <= cmi < hi:
            return name
    return bands[-1][2]


if __name__ == "__main__":
    test_cases = [
        (["EN", "EN", "EN", "EN"], "fully English -> CMI 0"),
        (["HI", "HI", "HI", "HI"], "fully Hindi -> CMI 0"),
        (["HI", "EN", "HI", "EN"], "50/50 mix -> CMI 50"),
        (["HI", "HI", "EN", "NE", "PUNCT"], "mostly Hindi with 1 EN + NE/punct excluded"),
        (["NE", "NUM", "PUNCT"], "fully language-independent -> CMI 0"),
        ([], "empty -> CMI 0"),
    ]
    for tags, desc in test_cases:
        print(f"{desc:55s} tags={tags!r:45s} CMI={compute_cmi(tags)}")