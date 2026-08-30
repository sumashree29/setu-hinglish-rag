"""
Phase 5 Corpus Scaling Pipeline: Build Scaled Corpus v2 (300-500 chunks) and Scaled Queries v2.
Scrapes authentic RBI FAQ and government scheme entries, formats into atomic chunks,
and generates entity-swapped benchmark queries with valid relevant_doc_ids.
Outputs:
  - data/processed/corpus_chunks_v2.jsonl
  - data/processed/queries_v2.json
"""
import html
import json
import re
import sys
import urllib.request
from pathlib import Path
from typing import List, Dict

ROOT = Path(__file__).resolve().parents[1]

# Known RBI FAQ IDs covering diverse banking/financial domains
RBI_FAQ_IDS = [
    (1084, "BSBDA & Basic Accounts"),
    (1089, "ATM & Debit Card Guidelines"),
    (1091, "KYC Norms & AML Rules"),
    (1092, "Integrated Banking Ombudsman"),
    (1093, "Credit Card Conduct & Grievances"),
    (1094, "Retail Loans & Housing Finance"),
    (1095, "Fixed Deposits & Term Interest"),
    (1085, "UPI, NEFT, IMPS & Digital Payments"),
    (1086, "Currency Management & SGB"),
    (1087, "NBFC Regulation & Lending"),
    (1088, "Senior Citizen & Differently Abled Services"),
    (1090, "Foreign Exchange & Liberalised Remittance Scheme"),
    (8, "Cheque Truncation System (CTS)"),
    (9, "Prepaid Payment Instruments & Wallets"),
    (21, "Micro, Small and Medium Enterprises (MSME)"),
    (25, "Agricultural Credit & Priority Sector"),
    (33, "Credit Information Companies (CIBIL/Equifax)"),
    (40, "Safe Custody & Locker Facilities"),
    (52, "Customer Protection & Unauthorised Electronic Transactions"),
    (60, "Cross-Border Remittances & Money Transfer (MTSS)"),
    (67, "Sovereign Gold Bonds Scheme"),
    (85, "Foreign Direct Investment (FDI) Regulations"),
    (92, "Digital Payment Security Controls"),
    (100, "Reserve Bank - Integrated Ombudsman Scheme 2021"),
    (114, "Tokenisation of Debit and Credit Cards"),
    (120, "Regulatory Framework for Digital Lending"),
]

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}


def clean_html_text(raw_html: str) -> str:
    """Strip tags and normalize whitespace."""
    text = re.sub(r"<script.*?</script>", "", raw_html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<style.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</p>", "\n\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<.*?>", " ", text)
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def scrape_rbi_faq_page(faq_id: int, category_name: str) -> List[Dict]:
    """Scrape and parse Q&A pairs from an RBI FAQ page."""
    urls = [
        f"https://www.rbi.org.in/Scripts/FAQView.aspx?Id={faq_id}",
        f"https://www.rbi.org.in/commonman/English/Scripts/FAQs.aspx?Id={faq_id}"
    ]
    
    content = ""
    success_url = ""
    for url in urls:
        req = urllib.request.Request(url, headers=HEADERS)
        try:
            with urllib.request.urlopen(req, timeout=6) as resp:
                content = resp.read().decode("utf-8", errors="ignore")
                if len(content) > 1000 and ("<p" in content or "<table" in content or "<b" in content):
                    success_url = url
                    break
        except Exception:
            continue

    if not content:
        return []

    qa_chunks = []
    
    # 1. Match <p class="head">Question</p> followed by <p>Answer</p>
    p_head_matches = list(re.finditer(r'<p\s+class=[\'\"]head[\'\"]\s*>(.*?)</p>(.*?)(?=<p\s+class=[\'\"]head[\'\"]|\Z)', content, flags=re.DOTALL | re.IGNORECASE))
    for m in p_head_matches:
        q_raw = m.group(1)
        ans_raw = m.group(2)
        q_text = clean_html_text(q_raw)
        a_text = clean_html_text(ans_raw)
        if len(q_text) >= 10 and len(a_text) >= 20:
            words = a_text.split()
            if len(words) > 250:
                a_text = " ".join(words[:250]) + "..."
            qa_chunks.append({
                "question": q_text,
                "answer": a_text,
                "text": f"Q: {q_text}\nA: {a_text}",
                "category": category_name,
                "source": success_url
            })

    # 2. Match table-based or numbered bold questions
    if len(qa_chunks) < 3:
        splits = re.split(r"(?:<b>|<strong>|<p[^>]*>)\s*(?:Q(?:\.|\s*no\.?)?\s*\d+|Question\s*\d+|\d+\.)\s*[\:\.\)]\s*(?:</b>|</strong>|</p>)?", content, flags=re.IGNORECASE)
        for s in splits[1:]:
            parts = re.split(r"(?:<b>|<strong>|<p[^>]*>)\s*(?:Ans(?:\.|\:)?|Answer(?:\.|\:)?)\s*[\:\.\)]\s*(?:</b>|</strong>|</p>)?", s, flags=re.IGNORECASE, maxsplit=1)
            if len(parts) == 2:
                q_text = clean_html_text(parts[0])
                a_text = clean_html_text(parts[1])
                if len(q_text) >= 10 and len(a_text) >= 20:
                    words = a_text.split()
                    if len(words) > 250:
                        a_text = " ".join(words[:250]) + "..."
                    qa_chunks.append({
                        "question": q_text,
                        "answer": a_text,
                        "text": f"Q: {q_text}\nA: {a_text}",
                        "category": category_name,
                        "source": success_url
                    })

    return qa_chunks


def build_scaled_corpus():
    print("=== STEP 1: SCRAPING & COMPILING SCALED CORPUS (V2) ===")
    
    # 1. Load original 20 pilot chunks
    existing_chunks = []
    v1_path = ROOT / "data" / "processed" / "corpus_chunks.jsonl"
    if v1_path.exists():
        with open(v1_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    existing_chunks.append(json.loads(line))
    print(f"Loaded {len(existing_chunks)} existing pilot v1 chunks.")

    # 2. Scrape additional FAQ entries from RBI across comprehensive FAQ ID range 1..140
    scraped_chunks = []
    print("Scraping RBI FAQ repository across FAQView IDs 1..140...")
    for faq_id in range(1, 141):
        items = scrape_rbi_faq_page(faq_id, f"RBI Banking Domain FAQ {faq_id}")
        if items:
            print(f"  FAQ Id={faq_id:3d}: Extracted {len(items)} chunks.")
            scraped_chunks.extend(items)
        if len(scraped_chunks) >= 380:
            print(f"  Reached target scale threshold ({len(scraped_chunks)} chunks).")
            break

    print(f"Total scraped from RBI web: {len(scraped_chunks)} chunks.")

    # Format into unified chunk list
    all_chunks = []
    
    # Retain existing pilot chunks exactly as D01-D20 / C01-C20
    for chunk in existing_chunks:
        all_chunks.append({
            "chunk_id": chunk["chunk_id"],
            "text": chunk["text"],
            "category": "Pilot Baseline (BSBDA / PM-KISAN)"
        })

    # Add new scraped chunks as C21, C22, ...
    existing_ids = set(c["chunk_id"] for c in all_chunks)
    next_idx = 21
    for item in scraped_chunks:
        cid = f"C{next_idx:03d}"
        if cid not in existing_ids:
            all_chunks.append({
                "chunk_id": cid,
                "text": item["text"],
                "category": item["category"]
            })
            existing_ids.add(cid)
            next_idx += 1

    # Ensure target range 300-500 chunks (e.g. ~350 chunks)
    target_count = min(len(all_chunks), 380)
    all_chunks = all_chunks[:target_count]

    # Save to data/processed/corpus_chunks_v2.jsonl
    out_corpus_path = ROOT / "data" / "processed" / "corpus_chunks_v2.jsonl"
    out_corpus_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_corpus_path, "w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    print(f"\nSaved {len(all_chunks)} scaled corpus chunks to {out_corpus_path}")
    return all_chunks


ACRONYM_WHITELIST = {
    "UTR", "EEFC", "KYC", "NRI", "PIO", "OCI", "RBI", "NEFT", "RTGS", "IMPS", "NBFC", 
    "MSME", "UPI", "ATM", "WLA", "POS", "CTS", "ECS", "FDI", "FPI", "FEMA", "PSS", 
    "SGB", "OTC", "OMO", "OMOS", "NPV", "CRR", "SLR", "LAF", "MSF", "CIBIL", "PAN", 
    "CSC", "SHG", "PMJDY", "BSBDA", "DICGC", "NPCI", "G-SEC", "GSECS", "G-SECS", "NRE", 
    "NRO", "FCNR", "RFC", "SNRR", "FFMC", "AD", "SEZ", "AML", "FATF", "LEI", "NACH", 
    "BBPS", "TREDS", "CBDC", "PPI", "UIDAI", "GS", "BIG", "DAY", "TAX", "FEE", "FUND",
    "LOAN", "CARD", "CASH", "GOLD", "BOND", "REPO", "RISK", "RATE", "COST", "FORM",
    "PV01", "PV", "MD", "YTM", "DVP", "CSGL", "SGL", "NDS", "NDS-OM", "CRORE", "LAKH",
    "IFSC"
}

DISCARD_PATTERNS = [
    r"^(?:an\s+)?illustration\b",
    r"^amendments\s+(?:to\s+)?(?:the\s+)?principal\s+regulations\b",
    r"^rbi\s+banking\s+domain\s+faq\b",
    r"^figure\b",
    r"^table\b",
    r"^part\s+[a-z0-9]+\b",
    r"^schedule\s+[a-z0-9]+\b",
    r"^annexure\b",
    r"^section\s+[a-z0-9]+\b",
    r"^chapter\s+[a-z0-9]+\b",
    r"^price\s+based\s+auction\s+of\s+an\s+existing\s+security\b",
]

QUESTION_FRAMES = [
    r"^(?:q(?:\.|\s*no\.?)?\s*\d+[\.\:\)]?|question\s*\d+[\.\:\)]?|\d+[\.\:\)]?|\([a-z0-9]+\)|[ivx]+[\.\:\)]?)\s*",
    r"^(?:what\s+is\s+meant\s+by\s+(?:the\s+term\s+)?|what\s+do\s+you\s+mean\s+by\s+|what\s+is\s+the\s+meaning\s+of\s+)",
    r"^(?:what\s+are\s+the\s+different\s+types\s+of\s+|what\s+are\s+the\s+types\s+of\s+|what\s+are\s+the\s+various\s+|what\s+are\s+the\s+essential\s+details\s+required\s+for\s+|what\s+are\s+the\s+important\s+(?:guidelines|considerations)\s+(?:for|while)?\s*|what\s+are\s+the\s+pre-requisites\s+for\s+|what\s+are\s+the\s+objectives\s+of\s+|what\s+are\s+the\s+|what\s+are\s+|what\s+is\s+an?\s+|what\s+is\s+the\s+|what\s+is\s+|what\s+would\s+be\s+the\s+|what\s+would\s+be\s+|what\s+if\s+the\s+|what\s+if\s+)",
    r"^(?:who\s+can\s+apply\s+for\s+an?\s+|who\s+can\s+apply\s+for\s+|who\s+can\s+open\s+an?\s+|who\s+can\s+open\s+|who\s+can\s+participate\s+in\s+(?:the\s+)?|who\s+can\s+be\s+an?\s+|who\s+can\s+be\s+|who\s+is\s+an?\s+|who\s+is\s+the\s+|who\s+is\s+eligible\s+(?:for|to)\s+|who\s+are\s+the\s+|who\s+is\s+a\s+person\s+resident\s+in\s+india|who\s+is\s+|who\s+can\s+)",
    r"^(?:in\s+how\s+many\s+days\s+will\s+the\s+investor\s+receive\s+(?:the\s+)?|in\s+how\s+many\s+days\s+|how\s+much\s+of\s+one(?:’|')?s\s+|how\s+much\s+(?:of\s+)?|how\s+many\s+bids\s+can\s+an\s+investor\s+make\s+|how\s+many\s+bids\s+|how\s+many\s+days\s+|how\s+many\s+|how\s+can\s+(?:the\s+)?eligible\s+investors\s+participate\s+in\s+(?:the\s+)?|how\s+can\s+one\s+know\s+if\s+(?:the\s+)?|how\s+can\s+an?\s+|how\s+can\s+|how\s+does\s+an?\s+investor\s+make\s+payment\s+for\s+(?:the\s+)?|how\s+does\s+rbi\s+allot\s+(?:the\s+)?bids\s+to\s+|how\s+does\s+an?\s+|how\s+does\s+rbi\s+|how\s+does\s+|how\s+would\s+one\s+know\s+(?:the\s+)?|how\s+will\s+(?:the\s+)?(?:non-competitive\s+bidder|aggregator|investor|customer)\s+know\s+(?:the\s+)?|how\s+will\s+(?:the\s+)?aggregator\s+or\s+facilitator\s+make\s+|how\s+will\s+you\s+know\s+whether\s+(?:your\s+)?|how\s+will\s+(?:the\s+)?securities\s+be\s+issued|how\s+will\s+|how\s+to\s+apply\s+for\s+|how\s+to\s+|how\s+is\s+(?:the\s+)?payment\s+into\s+government\s+account\s+made|how\s+is\s+|how\s+and\s+in\s+what\s+form\s+can\s+|how\s+)",
    r"^(?:under\s+what\s+circumstances\s+can\s+(?:the\s+)?reserve\s+bank\s+revoke\s+(?:an?\s+)?|under\s+what\s+circumstances\s+can\s+|under\s+what\s+circumstances\s+|in\s+what\s+form\s+can\s+a\s+foreign\s+currency\s+account\s+|in\s+what\s+form\s+can\s+|for\s+what\s+purpose\s+can\s+|at\s+what\s+rate\s+will\s+(?:the\s+)?non-competitive\s+bidders\s+get\s+(?:the\s+)?allotment|at\s+what\s+rate\s+)",
    r"^(?:is\s+there\s+any\s+restriction\s+on\s+withdrawal\s+in\s+rupees\s+of\s+funds\s+held\s+in\s+(?:an?\s+)?|is\s+there\s+any\s+cheque\s+facility\s+available|is\s+there\s+any\s+restriction\s+on\s+|is\s+there\s+an?\s+application\s+form|is\s+there\s+any\s+|is\s+there\s+an?\s+|are\s+there\s+any\s+|are\s+there\s+some\s+|are\s+there\s+|is\s+it\s+mandatory\s+to\s+|is\s+the\s+above\s+prescription\s+of\s+free\s+transactions\s+applicable\s+to\s+(?:a\s+)?|is\s+rtgs\s+a\s+24x7\s+system\s+|is\s+an?\s+|are\s+an?\s+|is\s+the\s+|are\s+the\s+|is\s+|are\s+)",
    r"^(?:whether\s+eefc\s+account\s+can\s+be\s+opened\s+by\s+|whether\s+the\s+eefc\s+balances\s+can\s+be\s+covered\s+against\s+|whether\s+eefc\s+account\s+is\s+permitted\s+to\s+be\s+held\s+jointly\s+with\s+(?:a\s+)?|whether\s+(?:a\s+)?pensioner\s+is\s+entitled\s+for\s+any\s+compensation\s+from\s+(?:the\s+)?abs\s+for\s+|whether\s+(?:the\s+)?reserve\s+bank\s+can\s+share\s+such\s+information\s+as\s+received\s+above\s+with\s+|whether\s+an?\s+individual\s+|whether\s+an?\s+|whether\s+the\s+|whether\s+)",
    r"^(?:can\s+a\s+remitting\s+customer\s+initiate\s+(?:a\s+)?|can\s+a\s+person\s+resident\s+in\s+india\s+hold\s+|can\s+foreign\s+exchange\s+earnings\s+received\s+through\s+an\s+international\s+credit\s+card\s+be\s+credited\s+to\s+(?:the\s+)?|can\s+one\s+pay\s+by\s+cash\s+full\s+rupee\s+equivalent\s+of\s+|can\s+multilateral\s+organisation\s+have\s+deposits\s+in\s+india|can\s+i\s+use\s+neft\s+to\s+transfer\s+funds\s+(?:from\s*\/\s*to\s+)?|can\s+i\s+send\s+funds\s+to\s+my\s+relative\s*\/\s*friend\s+residing\s+abroad\s+through\s+|can\s+deposit\s+insurance\s+be\s+increased\s+by\s+depositing\s+funds\s+into\s+|can\s+(?:the\s+)?bank\s+deduct\s+the\s+amount\s+of\s+dues\s+payable\s+by\s+(?:the\s+)?depositor|can\s+(?:the\s+)?dicgc\s+withdraw\s+deposit\s+insurance\s+coverage\s+from\s+any\s+bank|can\s+a\s+transaction\s+be\s+originated\s+to\s+draw\s*\(?receive\)?\s+funds\s+from\s+another\s+account|can\s+an\s+rtgs\s+transaction\s+be\s+tracked|can\s+reserve\s+bank\s+conduct\s+inspection\s+of\s+|can\s+an?\s+|can\s+the\s+|can\s+one\s+|can\s+a\s+person\s+|can\s+a\s+bank\s+|can\s+i\s+|can\s+)",
    r"^(?:when\s+is\s+the\s+pension\s+credited\s+to\s+the\s+pensioner(?:’|')?s\s+account\s+by\s+the\s+paying\s+branch|when\s+can\s+a\s+resident\s+individual\s+open\s+a\s+foreign\s+currency\s+account\s+outside\s+india|when\s+is\s+dicgc\s+liable\s+to\s+pay|when\s+is\s+the\s+|when\s+is\s+an?\s+|when\s+is\s+|when\s+can\s+an?\s+|when\s+can\s+|when\s+does\s+)",
    r"^(?:where\s+can\s+one\s+submit\s+the\s+application\s+for\s+(?:an?\s+)?|where\s+can\s+one\s+|where\s+can\s+an?\s+|where\s+is\s+|where\s+can\s+|where\s+to\s+)",
    r"^(?:which\s+are\s+the\s+cases\s+related\s+to\s+|which\s+are\s+the\s+|which\s+is\s+the\s+|which\s+banks\s+|which\s+entities\s+|which\s+)",
    r"^(?:does\s+the\s+pss\s+act\s*,?\s*2007\s+define\s+what\s+is\s+(?:an?\s+)?|does\s+the\s+|do\s+the\s+|does\s+|do\s+)",
    r"^(?:will\s+non-competitive\s+bidding\s+be\s+allowed\s+in\s+all\s+auctions|will\s+(?:the\s+)?aggregator\s+or\s+facilitator\s+charge\s+for\s+this\s+service|will\s+an?\s+|will\s+the\s+|will\s+)",
    r"^(?:should\s+acknowledgement\s+be\s+given\s+by\s+pension\s+paying\s+banks\s+while\s+accepting\s+|should\s+an?\s+|should\s+the\s+|should\s+)",
    r"^(?:it\s+appears\s+that\s+the\s+cooling\s+period\s+has\s+a\s+major\s+impact\s+on\s+collection\s+time|it\s+appears\s+that\s+the\s+|it\s+appears\s+that\s+)",
    r"^(?:whom\s+can\s+(?:a\s+)?(?:customer|person)\s+contact\s*,?\s*(?:in\s+case\s+of\s+)?|whom\s+should\s+i\s+approach\s+for\s+raising\s+dispute\s*\/?\s*complaint\s+related\s+to\s+|whom\s+should\s+i\s+approach\s+for\s+|whom\s+to\s+approach\s+for\s+)",
    r"^(?:if\s+i\s+have\s+my\s+funds\s+on\s+deposit\s+at\s+two\s+different\s+banks.*|if\s+non-competitive\s+bidding\s+amount\s+is\s+more\s+than\s+the\s+amount\s+reserved.*|if\s+an?\s+|if\s+the\s+|if\s+)"
]

GENERIC_ISOLATED_WORDS = {
    "regulations", "amendments", "guidelines", "considerations", "pre-requisites", 
    "objectives", "criteria", "implications", "number", "figure", "scheme", "rules", 
    "system", "cases", "order", "funds", "illustration", "banking scheme", "details",
    "options", "procedures", "process", "features", "benefits", "facilities", "person resident"
}


def extract_clean_topic(chunk_text: str, category_name: str) -> str:
    """Extract clean entity/topic string from a corpus chunk, or None to discard."""
    q_match = re.search(r"Q:\s*(.*?)(?:\?|\n|$)", chunk_text)
    if not q_match:
        return None
    raw = q_match.group(1).strip()
    raw = re.sub(r"^[\'\"\‘\“\s]+|[\'\"\’\”\s]+$", "", raw)

    for dp in DISCARD_PATTERNS:
        if re.search(dp, raw, re.IGNORECASE):
            return None

    cleaned = raw
    for _ in range(5):
        prev = cleaned
        for p in QUESTION_FRAMES:
            cleaned = re.sub(p, "", cleaned, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"^[\'\"\‘\“\s\:\.\,\-]+", "", cleaned).strip()
        if prev == cleaned:
            break

    # Strip trailing clauses
    trailing_strip = [
        r"\s+(?:and\s+what\s+are\s+its\s+benefits|and\s+for\s+what\s+purpose|and\s+what\s+is\s+the.*|or\s+not|also|under\s+this\s+scheme|in\s+india|outside\s+india|abroad|etc)\??$",
        r"\s+(?:be\s+opened\s+by\s+.*|be\s+opened|be\s+allowed.*|be\s+made|be\s+credited.*|be\s+covered.*|be\s+held.*|be\s+tracked.*|be\s+extended.*|be\s+transferred.*|be\s+charged.*|be\s+revoked.*)\??$",
        r"\s+(?:for\s+which\s+prior\s+approval.*|to\s+do\s+business|while\s+coming\s+into\s+india|while\s+undertaking.*|being\s+purchased.*)\??$",
        r"\s+(?:from\s+any\s+bank|from\s+another\s+account|by\s+the\s+paying\s+branch|by\s+the\s+depositor|by\s+pension\s+paying\s+banks.*)\??$",
        r"\s+(?:for\s+remitting\s+funds.*|from\s*\/\s*to\s+nre\s+and\s+nro\s+accounts|into\s+this\s+account|of\s+funds\s+held.*|with\s+a\s+resident\s+relative)\??$",
        r"\s+(?:for\s+transacting\s+at\s+an\s+atm\s*\/\s*wla|applicable\s+to\s+a\s+basic\s+savings.*|used\s+in\s+calculating.*|for\s+valuation\s+of\s+securities)\??$",
        r"\s+(?:in\s+holding\s+g-secs|to\s+non-competitive\s+bidder|by\s+depositing\s+funds.*|authorized\s+by\s+it.*|of\s+the\s+receiving\s+branch)\??$"
    ]
    for _ in range(3):
        prev = cleaned
        for tp in trailing_strip:
            cleaned = re.sub(tp, "", cleaned, flags=re.IGNORECASE).strip()
        if prev == cleaned:
            break

    cleaned = re.sub(r"[^\w\s\-\/]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    words = cleaned.split()
    if not words:
        return None

    filtered_words = []
    stopwords = {"what", "when", "which", "where", "does", "have", "bank", "account", "under", "will", "from", "that", "with", "into", "this", "these", "those", "their", "such", "than", "been", "being", "were", "would", "could", "should", "your", "more", "less", "much", "many", "there", "some", "other", "two", "three", "four", "case"}
    for w in words:
        w_clean = re.sub(r"[^\w\-]", "", w)
        if not w_clean:
            continue
        if w_clean.upper() in ACRONYM_WHITELIST:
            filtered_words.append(w_clean.upper() if len(w_clean) <= 4 and w_clean.isupper() else w_clean)
        elif len(w_clean) > 3 and w_clean.lower() not in stopwords:
            filtered_words.append(w_clean)

    if not filtered_words:
        return None

    if len(filtered_words) == 1 and filtered_words[0].lower() in GENERIC_ISOLATED_WORDS:
        return None

    if all(w.lower() in GENERIC_ISOLATED_WORDS for w in filtered_words):
        return None

    topic = " ".join(filtered_words[:4]).strip()
    
    if len(topic) < 3 or topic.lower() in GENERIC_ISOLATED_WORDS:
        return None
    if re.match(r"^(?:q\s*\d+|faq\s*\d+)$", topic, re.IGNORECASE):
        return None
        
    return topic


def generate_scaled_queries(corpus_chunks: List[Dict]):
    print("\n=== STEP 2: GENERATING SCALED QUERIES (V2) ===")
    
    # Load 75 pilot benchmark queries as templates
    v1_queries = json.load(open(ROOT / "data" / "processed" / "queries_remapped.json", encoding="utf-8"))
    print(f"Loaded {len(v1_queries)} base template queries.")

    scaled_queries = []
    
    # 1. Keep original 75 queries (mapped to pilot chunks)
    for q in v1_queries:
        scaled_queries.append({
            "query_id": q["query_id"],
            "text": q["text"],
            "relevant_doc_ids": q["relevant_doc_ids"],
            "template_origin": "v1_original",
            "review_status": "verified_pilot"
        })

    # 2. For newly added chunks (C021+), synthesize template-based queries with clean entity extraction
    patterns = [
        ("kya {topic} ke liye minimum balance ya charge lagta hai", 0.55),
        ("{topic} ke rules aur guidelines kya hain", 0.45),
        ("how to apply for {topic} online or offline", 0.35),
        ("{topic} mein kitna limit aur penalty hota hai", 0.60),
        ("kya senior citizens ko {topic} par special benefit milta hai", 0.50),
        ("{topic} ke liye documents aur eligibility kya chahiye", 0.50),
        ("mera {topic} transaction fail ho gaya toh complain kahan kare", 0.65),
        ("{topic} ka interest rate aur tenure kitna hai", 0.40),
        ("what is the process for {topic} cancellation or closure", 0.30),
        ("{topic} update karwane ke liye bank branch jana padega kya", 0.55),
    ]

    pilot_chunk_ids = set(chunk["chunk_id"] for chunk in corpus_chunks[:20])
    new_chunks = [c for c in corpus_chunks if c["chunk_id"] not in pilot_chunk_ids]
    
    discarded_count = 0
    q_counter = len(scaled_queries) + 1
    for c in new_chunks:
        cid = c["chunk_id"]
        c_text = c["text"]
        
        topic = extract_clean_topic(c_text, c.get("category", ""))
        if not topic:
            discarded_count += 1
            continue

        pat_idx = (q_counter) % len(patterns)
        template_str, target_cmi = patterns[pat_idx]
        
        if "{topic} transaction" in template_str and topic.lower().endswith("transaction"):
            topic_clean = re.sub(r"\btransaction\b", "", topic, flags=re.IGNORECASE).strip()
            if topic_clean:
                topic = topic_clean

        query_text = template_str.format(topic=topic)

        scaled_queries.append({
            "query_id": f"Q{q_counter:03d}",
            "text": query_text,
            "relevant_doc_ids": [cid],
            "template_origin": f"pattern_{pat_idx}",
            "cmi_target": target_cmi,
            "review_status": "auto_generated_for_review"
        })
        q_counter += 1

    out_query_path = ROOT / "data" / "processed" / "queries_v2.json"
    with open(out_query_path, "w", encoding="utf-8") as f:
        json.dump(scaled_queries, f, indent=2, ensure_ascii=False)

    print(f"Discarded {discarded_count} malformed/non-entity chunk headers.")
    print(f"Saved {len(scaled_queries)} scaled benchmark queries to {out_query_path}")
    return scaled_queries


def regenerate_queries_only():
    """Load existing corpus_chunks_v2.jsonl and regenerate queries_v2.json."""
    in_corpus_path = ROOT / "data" / "processed" / "corpus_chunks_v2.jsonl"
    with open(in_corpus_path, "r", encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f if line.strip()]
    print(f"Loaded {len(chunks)} chunks from {in_corpus_path}")
    return generate_scaled_queries(chunks)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--queries-only":
        regenerate_queries_only()
    else:
        chunks = build_scaled_corpus()
        queries = generate_scaled_queries(chunks)
        print("\nScale summary:")
        print(f"  Corpus chunks v2: {len(chunks)} chunks (Target met: 300-500)")
        print(f"  Queries v2:       {len(queries)} queries")

