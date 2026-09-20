import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
TABLES_DIR = ROOT / "results" / "tables"

def main():
    print("Running consistency check across final assets...")
    
    # 1. Load Baseline JSON
    baselines = json.load(open(TABLES_DIR / "scaled_corpus_retrieval_v3.json", "r", encoding="utf-8"))
    bge_mrr = baselines["bge_m3"]["overall_314_queries_380_chunks"]["mrr"]
    me5_mrr = baselines["me5_large"]["overall_314_queries_380_chunks"]["mrr"]
    
    # 2. Check RESULTS_SUMMARY.md
    summary_text = open(ROOT / "RESULTS_SUMMARY.md", "r", encoding="utf-8").read()
    
    if str(round(bge_mrr, 4)) not in summary_text:
        print(f"FAIL: BGE-M3 MRR {round(bge_mrr, 4)} not found in RESULTS_SUMMARY.md")
        return
        
    if str(round(me5_mrr, 4)) not in summary_text:
        print(f"FAIL: mE5-large MRR {round(me5_mrr, 4)} not found in RESULTS_SUMMARY.md")
        return
        
    print("PASS: Baseline metrics match in RESULTS_SUMMARY.md")
    
    # 3. Check overcorrection stats
    over = json.load(open(TABLES_DIR / "overcorrection_final.json", "r", encoding="utf-8"))
    holm_p = over["bge_m3"]["operators"]["LAG"]["wilcoxon_p_incorrect_holm"]
    
    if str(round(holm_p, 3)) not in summary_text:
        print(f"FAIL: Holm p-value {round(holm_p, 3)} not found in RESULTS_SUMMARY.md")
        return
        
    print("PASS: Overcorrection Holm p-value matches in RESULTS_SUMMARY.md")
    
    # 4. Check AUDIT_FINAL.md for limitation
    audit_text = open(ROOT / "AUDIT_FINAL.md", "r", encoding="utf-8").read()
    if "Phase 13 (CMI Validation)**: Out of scope" not in audit_text:
        print("FAIL: Phase 13 out-of-scope note not in AUDIT_FINAL.md")
        return
        
    print("PASS: AUDIT_FINAL.md out-of-scope notes verified.")
    print("\nALL CONSISTENCY CHECKS PASSED.")

if __name__ == "__main__":
    main()
