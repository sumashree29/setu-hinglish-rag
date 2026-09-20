import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TABLES_DIR = ROOT / "results" / "tables"
OUT_FILE = TABLES_DIR / "IEEE_TABLES.md"

def main():
    out = ["# IEEE Paper Tables\n"]
    
    # Table 1: Baseline Retrievers (Phase 9)
    try:
        baselines = json.load(open(TABLES_DIR / "scaled_corpus_retrieval_v3.json", "r", encoding="utf-8"))
        out.append("## Table I: Baseline Retriever Performance\n")
        out.append("| Model | MRR | NDCG@10 | Hit@10 |\n|---|---|---|---|")
        for m, res in baselines.items():
            if m == "summary": continue
            out.append(f"| {m} | {res.get('mrr', 0):.4f} | {res.get('ndcg_10', 0):.4f} | {res.get('hit_10', 0):.4f} |")
        out.append("\n")
    except Exception as e:
        out.append(f"*(Could not load baselines: {e})*\n")

    # Table 2: Operator Ablation (Phase 10)
    try:
        ablations = json.load(open(TABLES_DIR / "scaled_operator_ablation_v3.json", "r", encoding="utf-8"))
        out.append("## Table II: Operator Ablation (MRR)\n")
        out.append("| Embedding Model | RAW | +LQP | +CAEP | +LAG |\n|---|---|---|---|---|")
        for m, res in ablations.items():
            raw = res.get("RAW", {}).get("mrr", 0)
            lqp = res.get("LQP_alone", {}).get("mrr", 0)
            caep = res.get("CAEP_alone", {}).get("mrr", 0)
            lag = res.get("LAG_alone", {}).get("mrr", 0)
            out.append(f"| {m} | {raw:.4f} | {lqp:.4f} | {caep:.4f} | {lag:.4f} |")
        out.append("\n")
    except Exception as e:
        out.append(f"*(Could not load operator ablations: {e})*\n")
        
    # Table 3: Full Pipeline Comparison
    try:
        setu = json.load(open(TABLES_DIR / "setu_v1_v2_comparison_scaled.json", "r", encoding="utf-8"))
        out.append("## Table III: Full SETU Pipeline vs Baselines\n")
        out.append("| System | MRR | NDCG@10 | Hit@10 |\n|---|---|---|---|")
        for m, res in setu.items():
            out.append(f"| {m} | {res.get('mrr', 0):.4f} | {res.get('ndcg', 0):.4f} | {res.get('hit_ratio', 0):.4f} |")
        out.append("\n")
    except Exception as e:
        out.append(f"*(Could not load SETU comparison: {e})*\n")

    # Table 4: Over-correction Diagnosis
    try:
        over = json.load(open(TABLES_DIR / "overcorrection_final.json", "r", encoding="utf-8"))
        out.append("## Table IV: Over-Correction Diagnosis (Holm-corrected p-values)\n")
        out.append("| Model | Operator | ΔMRR (Correct) | p (Correct) | ΔMRR (Incorrect) | p (Incorrect) |\n|---|---|---|---|---|---|")
        for m, mres in over.items():
            if m == "phase4_chi2": continue
            ops = mres.get("operators", {})
            for op, opres in ops.items():
                dc = opres.get("mean_delta_correct", 0)
                pc = opres.get("wilcoxon_p_correct_holm", 1.0)
                di = opres.get("mean_delta_incorrect", 0)
                pi = opres.get("wilcoxon_p_incorrect_holm", 1.0)
                out.append(f"| {m} | {op} | {dc:.4f} | {pc:.4f} | {di:.4f} | {pi:.4f} |")
        out.append("\n")
    except Exception as e:
        out.append(f"*(Could not load overcorrection: {e})*\n")

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(out))
    print(f"Tables saved to {OUT_FILE}")

if __name__ == "__main__":
    main()
