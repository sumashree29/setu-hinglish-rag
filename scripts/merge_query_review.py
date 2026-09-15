import sys
import json
from pathlib import Path
from sklearn.metrics import cohen_kappa_score
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))
try:
    from setu.config import set_seed
    set_seed()
except ImportError:
    pass

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
REVIEW_DIR = DATA_DIR / "review"
RESULTS_DIR = ROOT / "results" / "tables"

def main():
    b1_path = REVIEW_DIR / "review_batch_1.json"
    b2_path = REVIEW_DIR / "review_batch_2.json"
    b3_path = REVIEW_DIR / "review_batch_3.json"
    
    if not (b1_path.exists() and b2_path.exists() and b3_path.exists()):
        print("Review batches not found. Run prepare_review_batches.py first, and have humans fill them out.")
        return
        
    with open(b1_path, "r", encoding="utf-8") as f:
        b1 = json.load(f)
    with open(b2_path, "r", encoding="utf-8") as f:
        b2 = json.load(f)
    with open(b3_path, "r", encoding="utf-8") as f:
        b3 = json.load(f)
        
    # Check overlap and calculate Kappa
    b1_dict = {q["query_id"]: q for q in b1}
    b2_dict = {q["query_id"]: q for q in b2}
    
    overlap_ids = set(b1_dict.keys()).intersection(set(b2_dict.keys()))
    print(f"Found {len(overlap_ids)} overlapping queries between batch 1 and 2.")
    assert len(overlap_ids) >= 20, "Overlap must be at least 20 queries!"
    
    # Calculate Cohen's Kappa for the boolean flags (y/n)
    # We will map 'y'->1, 'n'->0. If empty, we ignore or treat as 0, but they should be filled.
    def to_bin(val):
        return 1 if str(val).lower().strip() == 'y' else 0
        
    kappas = {}
    for field in ["is_fluent_hinglish", "is_answerable_from_corpus", "gold_doc_is_correct"]:
        r1_vals = [to_bin(b1_dict[qid].get(field)) for qid in overlap_ids]
        r2_vals = [to_bin(b2_dict[qid].get(field)) for qid in overlap_ids]
        k = cohen_kappa_score(r1_vals, r2_vals)
        kappas[field] = float(k) if not np.isnan(k) else 1.0
        
    agreement_file = RESULTS_DIR / "query_review_agreement.json"
    with open(agreement_file, "w", encoding="utf-8") as f:
        json.dump(kappas, f, indent=2)
    print(f"Saved agreement metrics to {agreement_file}")
    
    # Merge into queries_v3_final.json
    queries_file = DATA_DIR / "processed" / "queries_v3_final.json"
    with open(queries_file, "r", encoding="utf-8") as f:
        queries = json.load(f)
        
    # For overlap queries, just take reviewer 1's judgement (or resolve conflicts, but R1 is fine for now)
    reviewed_dict = {}
    for batch in [b1, b2, b3]:
        for q in batch:
            if q["query_id"] not in reviewed_dict:
                reviewed_dict[q["query_id"]] = q
                
    for q in queries:
        if q["query_id"] in reviewed_dict:
            rq = reviewed_dict[q["query_id"]]
            q["review_status"] = "human_verified"
            q["is_fluent_hinglish"] = (str(rq.get("is_fluent_hinglish")).lower().strip() == 'y')
            q["is_answerable_from_corpus"] = (str(rq.get("is_answerable_from_corpus")).lower().strip() == 'y')
            q["gold_doc_is_correct"] = (str(rq.get("gold_doc_is_correct")).lower().strip() == 'y')
            
            if rq.get("corrected_relevant_doc_ids"):
                q["relevant_doc_ids"] = rq["corrected_relevant_doc_ids"]
                
    # Overwrite queries file
    with open(queries_file, "w", encoding="utf-8") as f:
        json.dump(queries, f, indent=2)
        
    print(f"Merged review status into {queries_file}")

if __name__ == "__main__":
    main()
