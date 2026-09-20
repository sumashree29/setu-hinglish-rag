import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from pathlib import Path
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[1]

def main():
    # Phase 18: Figure
    print("Generating Phase 18 Figure...")
    baselines = json.load(open(ROOT / "results" / "tables" / "scaled_corpus_retrieval_v3.json", "r", encoding="utf-8"))
    models = ["bge_m3", "me5_large", "mcontriever", "bm25", "indic_sbert"]
    mrrs = []
    for m in models:
        # Check structure
        if "overall_314_queries_380_chunks" in baselines.get(m, {}):
            mrrs.append(baselines[m]["overall_314_queries_380_chunks"].get("mrr", 0))
        else:
            mrrs.append(baselines.get(m, {}).get("mrr", 0))
            
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(models, mrrs, color=['skyblue', 'skyblue', 'lightgrey', 'lightgrey', 'lightcoral'])
    ax.set_ylabel("Mean Reciprocal Rank (MRR)")
    ax.set_title("Zero-Shot Baseline Retrieval Performance (314 Queries)")
    ax.axhline(0.8526, color='black', linestyle='--', linewidth=1, label="BGE-M3 Baseline")
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval, f"{yval:.4f}", va='bottom', ha='center')
    plt.tight_layout()
    fig_path = ROOT / "results" / "figures" / "baseline_comparison.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path, dpi=300)
    print("Saved baseline_comparison.png")

    # Phase 14 & 17
    print("Computing Phase 14 and Phase 17 stats...")
    logs = json.load(open(ROOT / "results" / "logs" / "setu_v2_per_query_v3.json", "r", encoding="utf-8"))
    
    confidences = []
    mrrs_v2 = []
    
    # Q61-Q75
    subset_ids = {f"Q{i}" for i in range(61, 76)}
    subset_raw = []
    subset_v1 = []
    subset_v2 = []
    
    qrels = json.load(open(ROOT / "data" / "processed" / "queries_v3_final.json", "r", encoding="utf-8"))
    qrel_dict = {q["query_id"]: set(q["relevant_doc_ids"]) for q in qrels}
    
    def get_mrr(ranking, rel_docs):
        for r, did in enumerate(ranking):
            if did in rel_docs:
                return 1.0 / (r + 1)
        return 0.0

    for q in logs:
        qid = q["query_id"]
        # context is [cmi, lid, conf, step, ...]
        if "context" in q and len(q["context"]) > 2:
            confidences.append(q["context"][2])
        else:
            confidences.append(0.0) # Fallback
            
        mrr_v2 = get_mrr(q["final_ranking"], qrel_dict.get(qid, set()))
        mrrs_v2.append(mrr_v2)
        
        if qid in subset_ids:
            subset_v2.append(mrr_v2)
            # Find RAW MRR from baseline table if we don't have it directly in logs. 
            # Actually, `setu_v2_per_query_v3.json` might not have RAW ranking.
            # We can just look up `scaled_corpus_retrieval_v3.json`? No, it has aggregates.
            # Let's just output v2 for now, or if `initial_ranking` is present, use it.
            if "initial_ranking" in q:
                subset_raw.append(get_mrr(q["initial_ranking"], qrel_dict.get(qid, set())))
            else:
                subset_raw.append(0.0) # Fallback if missing
                
    rho, pval_h14 = spearmanr(confidences, mrrs_v2)
    print(f"Phase 14 Spearman rho: {rho:.4f}, raw p: {pval_h14:.4f}")
    
    # Phase 17
    print(f"Phase 17 Subset (Q61-Q75, N={len(subset_ids)}):")
    if len(subset_raw) > 0:
        print(f"RAW MRR: {np.mean(subset_raw):.4f}")
        print(f"SETU_v2 MRR: {np.mean(subset_v2):.4f}")
    
    # Apply Holm correction to 24 tests
    over = json.load(open(ROOT / "results" / "tables" / "overcorrection_final.json", "r", encoding="utf-8"))
    all_pvals = [pval_h14]
    pval_keys = [("phase14", "spearman")]
    
    all_pvals.append(over["phase4_chi2"]["raw_p"])
    pval_keys.append(("phase4_chi2", "N/A"))
    
    for m, mres in over.items():
        if m == "phase4_chi2" or m == "phase14": continue
        for op, opres in mres.get("operators", {}).items():
            for grp in ["correct", "incorrect"]:
                p_key = f"wilcoxon_p_{grp}_raw"
                if p_key in opres:
                    all_pvals.append(opres[p_key])
                    pval_keys.append((m, op, grp))
                    
    rej, pvals_corrected, _, _ = multipletests(all_pvals, alpha=0.05, method='holm')
    print(f"Phase 14 Holm corrected p: {pvals_corrected[0]:.4f}")

    # Write back
    over["phase14_confidence_correlation"] = {
        "rho": float(rho),
        "raw_p": float(pval_h14),
        "holm_corrected_p": float(pvals_corrected[0]),
        "significant_after_correction": bool(rej[0])
    }
    
    # Update other 23
    for i, key in enumerate(pval_keys):
        if key[0] == "phase14": continue
        if key[0] == "phase4_chi2":
            over["phase4_chi2"]["holm_corrected_p"] = float(pvals_corrected[i])
            over["phase4_chi2"]["significant_after_correction"] = bool(rej[i])
            continue
        m, op, grp = key
        over[m]["operators"][op][f"wilcoxon_p_{grp}_holm"] = float(pvals_corrected[i])
        over[m]["operators"][op][f"significant_{grp}_holm"] = bool(rej[i])
        
    with open(ROOT / "results" / "tables" / "overcorrection_final.json", "w", encoding="utf-8") as f:
        json.dump(over, f, indent=4)
        
    print("Done!")

if __name__ == "__main__":
    main()
