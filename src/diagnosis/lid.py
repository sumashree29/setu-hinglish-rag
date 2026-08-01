"""
LID — Token-level Language Identification tagger.

For Hinglish text (Latin-script Hindi mixed with English + Devanagari), we need
a per-token tag: HI, EN, NE (named entity), NUM, PUNCT, OTHER.

Design (CPU-only, no model download — good enough for pilot diagnosis in Phase 1;
can be swapped for a trained classifier later without changing the interface):

1. Devanagari script tokens -> HI directly (unambiguous).
2. Latin-script tokens -> checked against curated Hindi/English lexicons.
3. Capitalized unknown tokens -> NE (named entity).
4. Pure numeric tokens -> NUM.
5. Punctuation-only tokens -> PUNCT.
6. Anything else -> suffix heuristic, default fallback EN.

This is intentionally simple and transparent so it's easy to defend in an
interview or paper: "heuristic + lexicon baseline for Phase 1, upgradeable to
a trained token classifier in later phases."
"""
import re
import string
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
    if DEVANAGARI_RE.search(token):
        return "HI"

    stripped = token.strip(string.punctuation)
    if stripped == "":
        return "PUNCT"

    if NUM_RE.match(stripped):
        return "NUM"

    lower = stripped.lower()

    if lower in HINDI_LATIN_WORDS:
        return "HI"

    if lower in ENGLISH_COMMON_WORDS:
        return "EN"

    if stripped[0:1].isupper() and lower not in NE_EXCEPTIONS:
        return "NE"

    hindi_suffixes = ("na", "ne", "ta", "ti", "te", "ega", "egi", "enge", "gaya", "gayi")
    if lower.endswith(hindi_suffixes):
        return "HI"

    return "EN"


def tag_tokens(text: str) -> List[str]:
    tokens = text.split()
    return [_classify_token(tok) for tok in tokens]


def tag_pairs(text: str):
    tokens = text.split()
    return [(tok, _classify_token(tok)) for tok in tokens]


if __name__ == "__main__":
    samples = [
        "mera SBI account band karna hai kaise",
        "PM Kisan yojana ke liye online apply kaise kare",
        "What is the interest rate for RBI savings scheme",
        "Mujhe 50000 rupaye ka loan chahiye SBI se",
        "myScheme portal par registration kaise hota hai",
    ]
    for s in samples:
        print(f"\nText: {s}")
        for tok, tag in tag_pairs(s):
            print(f"  {tok:20s} -> {tag}")