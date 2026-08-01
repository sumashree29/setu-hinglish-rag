"""
Pilot corpus builder — scrape/chunk RBI + myScheme FAQ text, hand-write Hinglish
queries against it.
OWNER: R1 (query-writing split 3 ways per plan) | PHASE: 1 (plan §3.4), scaled in Phase 5 (§7.1)
"""
from pathlib import Path
from typing import List, Dict


def scrape_faq_source(url: str) -> List[str]:
    """
    Scrape a static FAQ page into a list of raw text blocks (one per Q&A or section).
    Sources: RBI (https://www.rbi.org.in/Scripts/FAQDisplay.aspx),
             myScheme (https://www.myscheme.gov.in/)

    TODO (R1):
        requests + BeautifulSoup, both sites are static HTML, no login needed.
        Fallback (plan §3.4 alt #1): if scraping is blocked/slow, manually
        paste 30-50 FAQ answers per site into data/raw/ instead and load from
        there — still enough for a 3,000-5,000-chunk pilot corpus.
    """
    raise NotImplementedError("R1: scrape or manually source FAQ text")


def chunk_text(text: str, chunk_size: int = 400) -> List[str]:
    """Split into ~300-500 token chunks (plan default: 400). Simple whitespace
    token count is fine — no need for a real tokenizer at pilot scale."""
    raise NotImplementedError("R1: implement chunking")


def build_pilot_corpus(n_queries: int = 100) -> Dict:
    """
    End-to-end: scrape both sources -> chunk -> save chunks to
    data/processed/corpus_chunks.jsonl, and leave a template spreadsheet/CSV
    at data/processed/pilot_queries_template.csv for the team to hand-write
    50-100 Hinglish queries against (spanning low to high code-mixing).

    TODO (R1 + team): this is the one task worth splitting 3 ways — once
    chunks exist, everyone can write ~15-30 queries each against different
    chunk ranges to move faster.
    """
    raise NotImplementedError("R1: wire up scrape_faq_source + chunk_text, save outputs")
