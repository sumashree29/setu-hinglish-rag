"""
Label pilot queries with LAG sub-strategies based on heuristic rules.

LIMITATION & METHODOLOGY NOTE (Plan §4.3):
Because empirical counterfactual search outcomes for all three rewrite sub-strategies
(light_normalize, dual_variant, full_rewrite) are not available a priori across all queries,
we apply a defensible rule-based labeling heuristic grounded in query diagnostic features:
  1. If entity_density > 0 AND the query text contains a lowercase/non-standard spelling
     of a known entity (verified via CAEP entity corrections on Q61-Q75) -> label 'full_rewrite' (2)
  2. If cmi_score >= 0.40 (high / very_high CMI bands per config.CMI_BANDS) AND no entity corruption
     -> label 'dual_variant' (1) to hedge against code-mixing degradation via multi-query fusion
  3. Otherwise (clean query text with low/medium CMI) -> label 'light_normalize' (0)
"""
import json
import pickle
import re
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path.append(str(root))

from setu.diagnosis.cmi import cmi
from setu.diagnosis.lid_entropy import lid_entropy
from setu.operators.caep import extract_entity_list
from setu.operators.lag import entity_density, SUB_STRATEGIES

# Canonical entities in this domain
CANONICAL_ENTITIES = ["BSBDA", "ATM", "Aadhaar", "KYC", "CSC", "PM-KISAN", "RBI"]

def has_corrupted_entity(query_text: str) -> bool:
    """Check if query contains misspelled or non-standard casing of known domain entities."""
    patterns = [
        r'\bbsbda\b', r'\bbsbdaa\b', r'\bbsb-da\b', r'\batm\b', r'\bkyc\b',
        r'\bpmkisan\b', r'\bpm-kisaan\b', r'\baadhar\b', r'\badhar\b', r'\bcsc\b',
        r'\bsmall acount\b', r'\bpasbook\b'
    ]
    for p in patterns:
        m = re.search(p, query_text)
        if m and m.group(0) not in CANONICAL_ENTITIES:
            return True
    return False

def main():
    chunks = [json.loads(line) for line in open(root / "data/processed/corpus_chunks.jsonl", encoding="utf-8")]
    doc_texts = [c["text"] for c in chunks]
    entities = extract_entity_list(doc_texts)
    
    queries = json.load(open(root / "data/processed/queries_remapped.json", encoding="utf-8"))
    
    labeled_queries = []
    strategy_counts = {s: 0 for s in SUB_STRATEGIES}
    
    for q in queries:
        qid = q["query_id"]
        text = q["text"]
        variant = q.get("variant", "unknown")
        
        cmi_score = cmi(text)
        entropy_score = lid_entropy(text)
        density = entity_density(text, entities)
        has_err = has_corrupted_entity(text)
        
        if density > 0 and has_err:
            strategy = "full_rewrite"
        elif cmi_score >= 0.40:
            strategy = "dual_variant"
        else:
            strategy = "light_normalize"
            
        strategy_counts[strategy] += 1
        label_idx = SUB_STRATEGIES.index(strategy)
        
        labeled_queries.append({
            "query_id": qid,
            "text": text,
            "variant": variant,
            "relevant_doc_ids": q.get("relevant_doc_ids", []),
            "cmi": float(cmi_score),
            "lid_entropy": float(entropy_score),
            "entity_density": float(density),
            "has_entity_error": bool(has_err),
            "strategy": strategy,
            "label": label_idx,
        })
        
    out_path = root / "data/processed/lag_labels.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(labeled_queries, f, indent=2, ensure_ascii=False)
        
    print(f"Saved {len(labeled_queries)} labeled queries -> {out_path}")
    print(f"Strategy Distribution: {strategy_counts}")

if __name__ == "__main__":
    main()
