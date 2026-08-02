"""
LID-entropy(q) — Shannon entropy over per-token language-ID predictions.
OWNER: R1 | PHASE: 1 (plan §3.5)

This module also owns per-token language tagging (load_lid_model, token_lid_tags),
shared by cmi.py so tagging only happens once per query.

CURRENT STATUS: token_lid_tags() uses a hand-curated lexicon + heuristic tagger
(hi/en/other) as a placeholder. Fine for the 60-query pilot corpus, but per the
SETU Implementation Plan §3.3/§3.5, this MUST be swapped for real IndicLID
(https://github.com/AI4Bharat/IndicLID) before Phase 5 scaling or final numbers.
Swapping only requires changing what's inside token_lid_tags() and
load_lid_model() -- callers (cmi.py, etc.) won't need to change.

TODO (R1, before Phase 5):
    - Download IndicLID model files (indiclid-ftn, indiclid-ftr, indiclid-bert)
      per https://github.com/AI4Bharat/IndicLID#download-indiclid-model
    - pip install fasttext
    - Replace load_lid_model()/token_lid_tags() internals with real IndicLID
      calls (see Inference/ai4bharat/IndicLID.py in that repo for the exact API)
"""
import re
import string
import math
from collections import Counter
from typing import List

DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
NUM_RE = re.compile(r"^[\d,.]+$")

HINDI_LATIN_WORDS = {
    "kya", "hai", "hain", "ka", "ke", "ki", "ko", "se", "mein", "me", "aur",
    "ya", "nahi", "nahin", "kaise", "kaun", "konsi", "kaunsa", "kitna", "kitne",
    "karna", "karne", "kare", "chahiye", "liye", "wala", "wale", "wali",
    "paisa", "paise", "rupaye", "khata", "khaata",
    "yojana", "yojna", "sarkar", "sarkari",
    "bharo", "bharna", "aavedan", "band",
    "khulwana", "kholna", "byaj",
    "milega", "milti", "milta", "hoga", "hogi", "hote",
    "kar", "sakte", "sakta", "sakti", "abhi", "turant",
    "mera", "meri", "mere", "mujhe", "mujhko", "hum", "humein", "tum",
    "hota", "hoti", "hue", "par", "iske", "uske", "yeh", "woh", "wagaira",
    "ek", "do", "teen", "char", "paanch", "main", "hoon", "khol",
    "khulwa", "khulti", "khulta", "sirf", "zyada", "kam", "bhi", "toh",
    "iska", "uska", "unka", "unki", "unke", "iski", "isse", "usse",
}

ENGLISH_COMMON_WORDS = {
    "the", "is", "are", "for", "to", "of", "in", "on", "and", "or", "what",
    "how", "when", "where", "can", "i", "my", "do", "does", "will", "would",
    "account", "bank", "loan", "interest", "scheme", "apply", "application",
    "online", "form", "close", "open", "government", "eligibility", "eligible",
    "documents", "required", "process", "status", "check", "balance",
    "registration", "portal", "rate", "savings", "myscheme",
}

NE_EXCEPTIONS = HINDI_LATIN_WORDS | ENGLISH_COMMON_WORDS


def _classify_token(token: str) -> str:
    """Returns 'hi', 'en', or 'other' (lowercase, matches IndicLID-style tags)."""
    if DEVANAGARI_RE.search(token):
        return "hi"

    stripped = token.strip(string.punctuation)
    if stripped == "":
        return "other"

    if NUM_RE.match(stripped):
        return "other"

    lower = stripped.lower()

    if lower in HINDI_LATIN_WORDS:
        return "hi"

    if lower in ENGLISH_COMMON_WORDS:
        return "en"

    if stripped[0:1].isupper() and lower not in NE_EXCEPTIONS:
        return "other"  # named entity -- language-independent bucket

    hindi_suffixes = ("na", "ne", "ta", "ti", "te", "ega", "egi", "enge", "gaya", "gayi")
    if lower.endswith(hindi_suffixes):
        return "hi"

    return "en"


def load_lid_model():
    """
    Load the LID tagger. Currently returns None (lexicon tagger needs no model
    object). Once swapped to real IndicLID, this will load and cache the
    IndicLID() object -- callers already pass this return value into
    token_lid_tags(), so no call-site changes will be needed when we swap.
    """
    return None


def token_lid_tags(query: str, model=None) -> List[str]:
    """
    Returns one language tag per token, e.g. ['hi','hi','en','hi'].
    `model` is accepted for interface-compatibility with the future real
    IndicLID version (unused by the current lexicon tagger).
    """
    tokens = query.split()
    return [_classify_token(tok) for tok in tokens]


def lid_entropy(query: str, model=None) -> float:
    """
    Shannon entropy (bits) over the per-token language-tag distribution.
    Higher = more balanced/unpredictable mixing. 0 = monolingual.
    """
    tags = token_lid_tags(query, model)
    n = len(tags)
    if n == 0:
        return 0.0

    counts = Counter(tags)
    entropy = 0.0
    for count in counts.values():
        p = count / n
        entropy -= p * math.log2(p)

    return round(entropy, 4)


if __name__ == "__main__":
    samples = [
        "mera SBI account band karna hai kaise",
        "What is the interest rate for RBI savings scheme",
        "Mujhe 50000 rupaye ka loan chahiye SBI se",
    ]
    for s in samples:
        tags = token_lid_tags(s)
        ent = lid_entropy(s)
        print(f"\n{s}\n  tags={tags}\n  entropy={ent}")