import sys
import json
import random
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
try:
    from setu.config import set_seed
    set_seed()
except ImportError:
    pass

import setu.config as config
from setu.diagnosis.cmi import cmi

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
REVIEW_DIR = DATA_DIR / "review"
REVIEW_DIR.mkdir(parents=True, exist_ok=True)

def cmi_band(score, bands):
    for lo, hi, name in bands:
        if lo <= score < hi:
            return name
    return bands[-1][2]

def main():
    queries_file = DATA_DIR / "processed" / "queries_v3_final.json"
    with open(queries_file, "r", encoding="utf-8") as f:
        queries = json.load(f)
        
    auto_queries = [q for q in queries if q.get("review_status") == "auto_generated_v3_paraphrased"]
    print(f"Found {len(auto_queries)} auto-generated paraphrased queries.")
    
    # Stratify by CMI band
    bands = {"low": [], "medium": [], "high": [], "very_high": []}
    for q in auto_queries:
        band = cmi_band(cmi(q["text"]), config.CMI_BANDS)
        bands[band].append(q)
        
    # We want 90 queries total, stratified proportionally.
    # Total = len(auto_queries)
    sampled = []
    set_seed()
    for b_name, b_queries in bands.items():
        n_samples = int(round(len(b_queries) / len(auto_queries) * 90))
        # ensure we don't sample more than available, just in case
        n_samples = min(n_samples, len(b_queries))
        sampled.extend(random.sample(b_queries, n_samples))
        
    # Adjust to exactly 90 if rounding was slightly off
    if len(sampled) < 90:
        remaining = 90 - len(sampled)
        pool = [q for q in auto_queries if q not in sampled]
        sampled.extend(random.sample(pool, remaining))
    elif len(sampled) > 90:
        sampled = sampled[:90]
        
    random.shuffle(sampled)
    
    # We need 3 batches of 30.
    # We also need 20 queries to overlap across two reviewers.
    # Let's make Reviewer 1 and Reviewer 2 share 20 queries.
    # So R1 gets queries 0-29.
    # R2 gets queries 10-29 (20 overlap) + 30-39 (10 new).
    # R3 gets queries 40-69.
    # Wait, total unique queries = 30 + 10 + 30 = 70. But we sampled 90!
    # Ah, if we have 90 unique queries, and we want 20 overlap, 
    # we need total items across all reviewers = 90 + 20 = 110.
    # But 3 batches of 30 = 90 slots.
    # If we put 90 unique queries into 90 slots, we have 0 overlap.
    # "Draw a stratified random sample of 90... 30 per reviewer. Have at least 20 queries overlap"
    # This implies 90 unique queries is NOT the pool size. 
    # It says "Draw a stratified random sample of 90... write it to review_batch_{1,2,3}.json - 30 per reviewer". 
    # 3 * 30 = 90. If they sum to 90, and 20 overlap, then unique queries = 70.
    # Let's draw 70 unique queries, duplicate 20 of them, total 90 records, split into 3 files of 30.
    
    # Let's adjust pool to 70 unique queries.
    sampled_70 = sampled[:70]
    
    overlap_20 = sampled_70[:20]
    unique_r1 = sampled_70[20:30]
    unique_r2 = sampled_70[30:40]
    unique_r3 = sampled_70[40:70]
    
    batch_1 = overlap_20 + unique_r1
    batch_2 = overlap_20 + unique_r2
    batch_3 = unique_r3
    
    assert len(batch_1) == 30
    assert len(batch_2) == 30
    assert len(batch_3) == 30
    
    def format_for_review(q_list):
        out = []
        for q in q_list:
            out.append({
                "query_id": q["query_id"],
                "text": q["text"],
                "source_question": q.get("source_question", ""),
                "is_fluent_hinglish": "",
                "is_answerable_from_corpus": "",
                "gold_doc_is_correct": "",
                "relevant_doc_ids": q["relevant_doc_ids"],
                "corrected_relevant_doc_ids": []
            })
        return out

    with open(REVIEW_DIR / "review_batch_1.json", "w", encoding="utf-8") as f:
        json.dump(format_for_review(batch_1), f, indent=2)
    with open(REVIEW_DIR / "review_batch_2.json", "w", encoding="utf-8") as f:
        json.dump(format_for_review(batch_2), f, indent=2)
    with open(REVIEW_DIR / "review_batch_3.json", "w", encoding="utf-8") as f:
        json.dump(format_for_review(batch_3), f, indent=2)
        
    print("Generated data/review/review_batch_{1,2,3}.json")

if __name__ == "__main__":
    main()
