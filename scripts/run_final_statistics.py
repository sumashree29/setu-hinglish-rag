import json
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr, wilcoxon
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[1]

def get_mrr(ranking, rel_docs):
    for r, did in enumerate(ranking):
        if did in rel_docs:
            return 1.0 / (r + 1)
    return 0.0

def main():
    queries = json.load(open(ROOT / "data" / "processed" / "queries_v3_final.json", "r", encoding="utf-8"))
    qrel_dict = {q["query_id"]: set(q["relevant_doc_ids"]) for q in queries}
    cmi_dict = {q["query_id"]: q.get("cmi", 0.0) for q in queries}

    per_query = []
    with open(ROOT / "results" / "canonical" / "per_query_results.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            per_query.append(json.loads(line))
            
    # Over-correction arrays
    raw_mrr = []
    v1_mrr = []
    v2_mrr = []
    cmis = []
    
    # Conditional split
    correct_v2 = []
    incorrect_v2 = []
    correct_raw = []
    incorrect_raw = []
    
    for q in per_query:
        qid = q["query_id"]
        rel_docs = qrel_dict.get(qid, set())
        
        m_raw = get_mrr(q["raw_ranking"], rel_docs)
        m_v1 = get_mrr(q["v1_ranking"], rel_docs)
        m_v2 = get_mrr(q["v2_ranking"], rel_docs)
        
        raw_mrr.append(m_raw)
        v1_mrr.append(m_v1)
        v2_mrr.append(m_v2)
        cmis.append(cmi_dict.get(qid, 0.0))
        
        # We define "correct" as MRR == 1.0 for strict conditional analysis, or MRR > 0.
        # Following Phase 11, it was MRR == 1.0. Let's use MRR == 1.0
        if m_raw == 1.0:
            correct_raw.append(m_raw)
            correct_v2.append(m_v2)
        else:
            incorrect_raw.append(m_raw)
            incorrect_v2.append(m_v2)
            
    raw_mrr = np.array(raw_mrr)
    v1_mrr = np.array(v1_mrr)
    v2_mrr = np.array(v2_mrr)
    
    aggregate = {
        "RAW_MRR": float(np.mean(raw_mrr)),
        "SETU_v1_MRR": float(np.mean(v1_mrr)),
        "SETU_v2_MRR": float(np.mean(v2_mrr)),
        "Queries": len(per_query)
    }
    with open(ROOT / "results" / "canonical" / "aggregate_results.json", "w", encoding="utf-8") as f:
        json.dump(aggregate, f, indent=4)
        
    pvals = []
    keys = []
    
    # H1: CMI vs RAW MRR
    rho_h1, p_h1 = spearmanr(cmis, raw_mrr)
    pvals.append(p_h1)
    keys.append("H1_CMI_vs_RAW_MRR")
    
    # H8: v1 vs v2 steps
    step_data = json.load(open(ROOT / "results" / "canonical" / "step_analysis.json", "r"))
    pvals.append(step_data["comparison"]["wilcoxon_p"])
    keys.append("H8_v1_vs_v2_steps")
    
    # H10: Confidence vs Final MRR
    conf_data = json.load(open(ROOT / "results" / "canonical" / "confidence_analysis.json", "r"))
    # The confidence analysis returned NaN for p-value because variance is zero.
    p_h10 = conf_data["statistics"]["spearman_p"]
    if p_h10 is not None and not np.isnan(p_h10):
        pvals.append(p_h10)
        keys.append("H10_Confidence_vs_MRR")
        
    # v1 vs v2 MRR (paired test)
    stat_v1_v2, p_v1_v2 = wilcoxon(v1_mrr, v2_mrr)
    pvals.append(p_v1_v2)
    keys.append("SETU_v1_vs_v2_MRR")
    
    # Overcorrection (v2 vs RAW on correct queries)
    if len(correct_raw) > 0:
        stat_oc, p_oc = wilcoxon(correct_raw, correct_v2)
        pvals.append(p_oc)
        keys.append("Overcorrection_Degradation_on_Correct")
        
    # Lift (v2 vs RAW on incorrect queries)
    if len(incorrect_raw) > 0:
        stat_lift, p_lift = wilcoxon(incorrect_raw, incorrect_v2)
        pvals.append(p_lift)
        keys.append("Lift_on_Incorrect")
        
    # Apply Holm
    # We must filter out nans if any
    valid_pvals = [p for p in pvals if p is not None and not np.isnan(p)]
    rej, corr_pvals, _, _ = multipletests(valid_pvals, alpha=0.05, method='holm')
    
    results = {}
    valid_idx = 0
    for key, raw_p in zip(keys, pvals):
        if raw_p is None or np.isnan(raw_p):
            results[key] = {"raw_p": None, "holm_p": None, "significant": False}
        else:
            results[key] = {
                "raw_p": float(raw_p),
                "holm_p": float(corr_pvals[valid_idx]),
                "significant": bool(rej[valid_idx])
            }
            valid_idx += 1
            
    # Add context data
    results["Overcorrection_Context"] = {
        "n_correct_queries": len(correct_raw),
        "mean_delta_mrr": float(np.mean(np.array(correct_v2) - np.array(correct_raw))) if len(correct_raw) > 0 else 0
    }
    results["Lift_Context"] = {
        "n_incorrect_queries": len(incorrect_raw),
        "mean_delta_mrr": float(np.mean(np.array(incorrect_v2) - np.array(incorrect_raw))) if len(incorrect_raw) > 0 else 0
    }
    results["SETU_v1_vs_v2_Context"] = {
        "v1_mean_mrr": aggregate["SETU_v1_MRR"],
        "v2_mean_mrr": aggregate["SETU_v2_MRR"],
        "mean_delta": aggregate["SETU_v2_MRR"] - aggregate["SETU_v1_MRR"]
    }
    
    with open(ROOT / "results" / "canonical" / "statistical_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
        
    print(f"Stats saved to results/canonical/statistical_results.json")

if __name__ == "__main__":
    main()
