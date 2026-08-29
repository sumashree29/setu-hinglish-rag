"""
Pilot corpus builder -- scrape/chunk RBI + myScheme FAQ text, hand-write Hinglish
queries against it.
OWNER: R1 (query-writing split 3 ways per plan) | PHASE: 1 (plan §3.4), scaled in Phase 5 (§7.1)

STATUS: Corpus built with 20 atomic chunks at data/processed/corpus_chunks.jsonl
(RBI BSBDA FAQ + PM-KISAN scheme content), with 75 hand-written Hinglish
queries at data/processed/queries_remapped.json (60 original + 15 misspelled-entity
subset). scrape_faq_source() and chunk_text() below are left as NotImplementedError
since the automated scrape path was not needed for pilot scale -- build_pilot_corpus()
instead just loads the already-built files. Real scraping would only be needed for
Phase 5 scaling.
"""
import json
from pathlib import Path
from typing import List, Dict


def scrape_faq_source(url: str) -> List[str]:
    """
    NOT IMPLEMENTED for pilot scale -- corpus was built via manual-paste
    fallback (plan §3.4 alt #1) instead of scraping. Implement this with
    requests + BeautifulSoup if/when Phase 5 requires automated scraping
    to hit the 3,000-5,000 chunk target.
    """
    raise NotImplementedError(
        "R1: not needed at pilot scale (manual-paste fallback used instead); "
        "implement with requests+BeautifulSoup for Phase 5 scaling"
    )


def chunk_text(text: str, chunk_size: int = 400) -> List[str]:
    """
    NOT IMPLEMENTED for pilot scale -- chunks were hand-merged to ~250-280
    words instead of programmatically split. Implement simple whitespace-token
    chunking here for Phase 5 scaling.
    """
    raise NotImplementedError(
        "R1: not needed at pilot scale (chunks hand-merged instead); "
        "implement whitespace-token chunking for Phase 5 scaling"
    )


def build_pilot_corpus(n_queries: int = 100) -> Dict:
    """
    Loads the already-built pilot corpus (20 atomic chunks, 75 queries) rather
    than rebuilding it from scratch. Returns both as a dict for convenience.
    """
    root = Path(__file__).resolve().parents[2]

    chunks = []
    with open(root / "data" / "processed" / "corpus_chunks.jsonl", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))

    queries = json.load(open(root / "data" / "processed" / "queries_remapped.json", encoding="utf-8"))

    return {"chunks": chunks, "queries": queries}


if __name__ == "__main__":
    corpus = build_pilot_corpus()
    print(f"Loaded {len(corpus['chunks'])} chunks, {len(corpus['queries'])} queries")
    for c in corpus["chunks"]:
        print(f"  {c['chunk_id']}: {len(c['text'].split())} words")