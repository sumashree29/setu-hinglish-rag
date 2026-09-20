import json
import sys
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]

def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def read_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def main():
    errors = []
    
    try:
        tables = read_json(ROOT / "results" / "tables" / "ieee_ready_tables.json")
    except Exception as e:
        print(f"FAILED TO LOAD IEEE TABLES: {e}")
        sys.exit(1)
        
    canonical_raw_mrr = f"{tables['Table_2_Baseline']['mrr']:.4f}"
    
    h6_verdict = tables["Table_4_Hypotheses"].get("H6", {}).get("verdict", "").lower()
    
    # Documents to check
    doc_paths = [
        "RESULTS_SUMMARY.md",
        "README.md",
        "CLAIM_CONTRACT.md",
        "CLAIM_EVIDENCE_MATRIX.md",
        "FINAL_EVIDENCE_REPORT.md",
        "FINAL_EVIDENCE_SCOPE.md"
    ]
    
    for name in doc_paths:
        try:
            content = read_file(ROOT / name)
        except Exception:
            errors.append(f"File {name} not found.")
            continue
            
        if "TBD" in content:
            errors.append(f"{name} contains 'TBD'")
            
        lower_content = content.lower()
        if "statistically equivalent" in lower_content or "statistical equivalence" in lower_content:
             errors.append(f"{name} claims 'statistically equivalent' without TOST/equivalence margin.")
             
        if name == "RESULTS_SUMMARY.md" and canonical_raw_mrr not in content:
            errors.append(f"{name} is missing the exact canonical RAW MRR ({canonical_raw_mrr})")

        # Check for claims of causality from correlation
        if "proves causality" in lower_content or "causes retrieval" in lower_content:
            errors.append(f"{name} contains unsupported causal claims.")
            
        # Check for absolute controller independence claims
        if "completely independent" in lower_content:
            errors.append(f"{name} contains exaggerated 'completely independent' controller claims.")
            
    if errors:
        print("CONSISTENCY ERRORS FOUND:")
        for e in errors:
            print(f" - {e}")
        sys.exit(1)
        
    print("ALL CONSISTENCY CHECKS PASSED.")
    sys.exit(0)

if __name__ == "__main__":
    main()
