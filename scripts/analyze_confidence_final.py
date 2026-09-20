import json
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]

def get_mrr(ranking, rel_docs):
    for r, did in enumerate(ranking):
        if did in rel_docs:
            return 1.0 / (r + 1)
    return 0.0

def main():
    queries = json.load(open(ROOT / "data" / "processed" / "queries_v3_final.json", "r", encoding="utf-8"))
    qrel_dict = {q["query_id"]: set(q["relevant_doc_ids"]) for q in queries}

    per_query = []
    with open(ROOT / "results" / "canonical" / "per_query_results.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            per_query.append(json.loads(line))
            
    analysis_logs = []
    initial_confs = []
    final_mrrs = []
    
    for q in per_query:
        qid = q["query_id"]
        rel_docs = qrel_dict.get(qid, set())
        
        conf_trace = q["v2_confidence_trace"]
        init_conf = conf_trace[0] if conf_trace else 0.0
        final_conf = conf_trace[-1] if conf_trace else 0.0
        
        init_mrr = get_mrr(q["raw_ranking"], rel_docs)
        final_mrr = get_mrr(q["v2_ranking"], rel_docs)
        steps = q["v2_n_steps"]
        
        initial_confs.append(init_conf)
        final_mrrs.append(final_mrr)
        
        analysis_logs.append({
            "query_id": qid,
            "initial_confidence": init_conf,
            "final_confidence": final_conf,
            "initial_mrr": init_mrr,
            "final_mrr": final_mrr,
            "step_count": steps
        })
        
    initial_confs = np.array(initial_confs)
    final_mrrs = np.array(final_mrrs)
    
    # Calculate Spearman
    rho, pval = spearmanr(initial_confs, final_mrrs)
    
    # Analyze signal distribution
    variance = np.var(initial_confs)
    unique_vals = len(np.unique(initial_confs))
    percentiles = {
        "p0": np.min(initial_confs),
        "p25": np.percentile(initial_confs, 25),
        "p50": np.percentile(initial_confs, 50),
        "p75": np.percentile(initial_confs, 75),
        "p99": np.percentile(initial_confs, 99),
        "p100": np.max(initial_confs)
    }
    
    # 0.2 was the gating threshold in v1. See how many are below it.
    below_0_2 = np.sum(initial_confs < 0.2)
    
    out = {
        "logs": analysis_logs,
        "statistics": {
            "spearman_rho": float(rho) if not np.isnan(rho) else None,
            "spearman_p": float(pval) if not np.isnan(pval) else None,
            "variance": float(variance),
            "unique_values_count": int(unique_vals),
            "total_queries": len(initial_confs),
            "queries_below_0_2": int(below_0_2),
            "distribution": {k: float(v) for k, v in percentiles.items()}
        }
    }
    
    with open(ROOT / "results" / "canonical" / "confidence_analysis.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=4)
        
    print(f"Confidence analysis saved. Variance: {variance:.4f}. Below 0.2: {below_0_2}/{len(initial_confs)}")

if __name__ == "__main__":
    main()
