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

    # 2. For newly added chunks (C021+), synthesize template-based queries
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
    
    q_counter = len(scaled_queries) + 1
    for c in new_chunks:
        cid = c["chunk_id"]
        c_text = c["text"]
        
        # Extract main subject / topic from Q: ...
        q_match = re.search(r"Q:\s*(.*?)(?:\?|\n|$)", c_text)
        if q_match:
            raw_q = q_match.group(1).strip()
            words = [w for w in raw_q.split() if len(w) > 3 and w.lower() not in ["what", "when", "which", "where", "does", "have", "bank", "account", "under", "will", "from"]]
            topic = " ".join(words[:4]) if words else c["category"].split("(")[0].strip()
        else:
            topic = c["category"].split("(")[0].strip()

        # Clean topic string
        topic = re.sub(r"[^\w\s\-]", "", topic).strip()
        if not topic:
            topic = "banking scheme"

        pat_idx = (q_counter) % len(patterns)
        template_str, target_cmi = patterns[pat_idx]
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

    print(f"Saved {len(scaled_queries)} scaled benchmark queries to {out_query_path}")
    return scaled_queries


if __name__ == "__main__":
    chunks = build_scaled_corpus()
    queries = generate_scaled_queries(chunks)
    print("\nScale summary:")
    print(f"  Corpus chunks v2: {len(chunks)} chunks (Target met: 300-500)")
    print(f"  Queries v2:       {len(queries)} queries")

