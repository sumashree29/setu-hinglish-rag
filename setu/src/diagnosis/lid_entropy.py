"""
LID-Entropy — Shannon entropy over token-level language tags.

Complements CMI: two utterances can have identical CMI but very different
switching patterns. E.g.:
    ["HI","HI","HI","EN","EN","EN"]  -> one clean switch point
    ["HI","EN","HI","EN","HI","EN"]  -> rapid alternation
Both have the same language-count ratio (so same CMI), but the second is far
more "unpredictable" -- that's what entropy captures.

We report two versions:
  1. Distributional entropy: Shannon entropy over the *proportion* of each
     language tag (ignores order/switching, just proportions).
  2. Switch-rate: fraction of adjacent token pairs where the tag changes.
     (Not entropy in the strict sense, but a cheap complementary "switching
     frequency" signal that's easy to explain in a paper/interview.)
"""
import math
from collections import Counter
from typing import List


def distributional_entropy(tags: List[str]) -> float:
    """
    Shannon entropy (base 2) over the language-tag distribution.
    Higher = more balanced mix of languages. 0 = monolingual.
    """
    n = len(tags)
    if n == 0:
        return 0.0

    counts = Counter(tags)
    entropy = 0.0
    for count in counts.values():
        p = count / n
        entropy -= p * math.log2(p)

    return round(entropy, 4)


def switch_rate(tags: List[str]) -> float:
    """
    Fraction of adjacent token pairs where the language tag changes.
    0 = no switching (monolingual or one single block).
    1 = switches at every single token boundary.
    """
    if len(tags) < 2:
        return 0.0

    switches = sum(1 for i in range(len(tags) - 1) if tags[i] != tags[i + 1])
    return round(switches / (len(tags) - 1), 4)


if __name__ == "__main__":
    test_cases = [
        (["HI", "HI", "HI", "EN", "EN", "EN"], "one clean switch block"),
        (["HI", "EN", "HI", "EN", "HI", "EN"], "rapid alternation"),
        (["HI", "HI", "HI", "HI"], "monolingual Hindi"),
        (["EN", "EN", "EN", "EN"], "monolingual English"),
        (["HI", "EN", "NE", "NUM"], "mixed with NE/NUM"),
        ([], "empty"),
    ]
    for tags, desc in test_cases:
        ent = distributional_entropy(tags)
        sw = switch_rate(tags)
        print(f"{desc:30s} tags={tags!r:45s} entropy={ent:.4f}  switch_rate={sw:.4f}")