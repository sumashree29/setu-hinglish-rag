import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TABLES_DIR = ROOT / "results" / "tables"
OUT_MD = TABLES_DIR / "IEEE_TABLES.md"
OUT_TEX = ROOT / "results" / "figures" / "ieee_tables.tex"

def main():
    out = ["# IEEE Paper Tables (Canonically Generated)\n"]
    tex_out = ["% Auto-generated IEEE Tables\n"]
    
    # Table 1: Baseline Retrievers (Phase 9)
    try:
        baselines = json.load(open(TABLES_DIR / "scaled_corpus_retrieval_v3.json", "r", encoding="utf-8"))
        out.append("## Table I: Baseline Retriever Performance (Full 314-Query / 380-Chunk Corpus)\n")
        out.append("| Model | Architecture | MRR | NDCG@10 | Hit@1 | Recall@10 |")
        out.append("|---|---|---|---|---|---|")
        
        arch_map = {
            "bge_m3": "Dense Multi-Lingual (XLM-RoBERTa)",
            "me5_large": "Dense Multi-Lingual (mE5)",
            "mcontriever": "Dense Multi-Lingual (Contriever)",
            "indic_sbert": "Dense Indic-Tuned (MuRIL)",
            "bm25": "Sparse Lexical (BM25)"
        }
        
        tex_out.append("\\begin{table}[t]\n\\caption{Baseline Retriever Performance on 314 Hinglish Queries (380-Chunk Corpus)}\n\\label{tab:baselines}\n\\centering\n\\begin{tabular}{lcccc}\n\\hline\nModel & MRR & NDCG@10 & Hit@1 & Recall@10 \\\\\n\\hline")
        
        for m, res in baselines.items():
            if m == "summary" or not isinstance(res, dict): continue
            overall = res.get("overall_314_queries_380_chunks", {})
            mrr = overall.get("mrr", 0.0)
            ndcg10 = overall.get("ndcg@10", 0.0)
            hit1 = overall.get("hit_rate@1", 0.0)
            rec10 = overall.get("recall@10", 0.0)
            arch = arch_map.get(m, "Dense")
            out.append(f"| **{m}** | {arch} | {mrr:.4f} | {ndcg10:.4f} | {hit1:.4f} | {rec10:.4f} |")
            tex_out.append(f"\\textbf{{{m}}} & {mrr:.4f} & {ndcg10:.4f} & {hit1:.4f} & {rec10:.4f} \\\\")
            
        out.append("\n")
        tex_out.append("\\hline\n\\end{tabular}\n\\end{table}\n\n")
    except Exception as e:
        out.append(f"*(Could not load baselines: {e})*\n")

    # Table 2: Operator Ablation (Phase 10)
    try:
        ablations = json.load(open(TABLES_DIR / "scaled_operator_ablation_v3.json", "r", encoding="utf-8"))
        out.append("## Table II: Operator Ablation (MRR on Full 314 Queries)\n")
        out.append("| Embedding Model | RAW Baseline | +LQP Alone | +CAEP Alone | +LAG Alone |")
        out.append("|---|---|---|---|---|")
        
        tex_out.append("\\begin{table}[t]\n\\caption{Standalone Operator Ablation (MRR)}\n\\label{tab:ablation}\n\\centering\n\\begin{tabular}{lcccc}\n\\hline\nEmbedding Model & RAW & +LQP & +CAEP & +LAG \\\\\n\\hline")
        
        for m, res in ablations.items():
            if not isinstance(res, dict): continue
            overall = res.get("overall_314", {})
            raw = overall.get("RAW", {}).get("mrr", 0.0)
            lqp = overall.get("LQP_alone", {}).get("mrr", 0.0)
            caep = overall.get("CAEP_alone", {}).get("mrr", 0.0)
            lag = overall.get("LAG_alone", {}).get("mrr", 0.0)
            out.append(f"| **{m}** | {raw:.4f} | {lqp:.4f} | {caep:.4f} | {lag:.4f} |")
            tex_out.append(f"\\textbf{{{m}}} & {raw:.4f} & {lqp:.4f} & {caep:.4f} & {lag:.4f} \\\\")
            
        out.append("\n")
        tex_out.append("\\hline\n\\end{tabular}\n\\end{table}\n\n")
    except Exception as e:
        out.append(f"*(Could not load operator ablations: {e})*\n")
        
    # Table 3: Full Pipeline Comparison
    try:
        ieee_ready_path = TABLES_DIR / "ieee_ready_tables.json"
        if ieee_ready_path.exists():
            ieee_ready = json.load(open(ieee_ready_path, "r", encoding="utf-8"))
            full_data = ieee_ready.get("Table_3_RAW_v1_v2", {})
        else:
            setu = json.load(open(TABLES_DIR / "setu_v1_v2_comparison_scaled.json", "r", encoding="utf-8"))
            full_data = setu.get("full_dataset", {})
        out.append("## Table III: Full SETU Pipeline vs Baselines (BGE-M3 Backbone)\n")
        out.append("| Configuration | Description | MRR | NDCG@10 | Recall@10 | Mean Actions | Latency (ms) |")
        out.append("|---|---|---|---|---|---|---|")
        
        tex_out.append("\\begin{table}[t]\n\\caption{Full SETU Pipeline vs Baselines (BGE-M3 Backbone)}\n\\label{tab:full_setu}\n\\centering\n\\begin{tabular}{lccccc}\n\\hline\nSystem & MRR & NDCG@10 & Recall@10 & Actions & Latency (ms) \\\\\n\\hline")
        
        desc_map = {
            "RAW": "Zero-shot dense baseline (no modification)",
            "SETU_v1": "Fixed pipeline (LAG -> CAEP -> LQP -> CARF)",
            "SETU_v2": "Adaptive contextual bandit controller (LinUCB)"
        }
        
        for cfg_name in ["RAW", "SETU_v1", "SETU_v2"]:
            cfg = full_data.get(cfg_name, {})
            mrr = cfg.get("mrr", 0.0)
            ndcg = cfg.get("ndcg@10", 0.0)
            rec10 = cfg.get("recall@10", 0.0)
            steps = cfg.get("mean_steps", 0.0)
            lat = cfg.get("mean_latency_ms", 0.0)
            desc = desc_map.get(cfg_name, "")
            steps_str = f"{steps:.3f}" if steps > 0 else "-"
            out.append(f"| **{cfg_name}** | {desc} | {mrr:.4f} | {ndcg:.4f} | {rec10:.4f} | {steps_str} | {lat:.1f} |")
            tex_out.append(f"\\textbf{{{cfg_name}}} & {mrr:.4f} & {ndcg:.4f} & {rec10:.4f} & {steps_str} & {lat:.1f} \\\\")
            
        out.append("\n")
        tex_out.append("\\hline\n\\end{tabular}\n\\end{table}\n\n")
    except Exception as e:
        out.append(f"*(Could not load SETU comparison: {e})*\n")

    # Table 4: Over-correction Diagnosis
    try:
        over = json.load(open(TABLES_DIR / "overcorrection_final.json", "r", encoding="utf-8"))
        out.append("## Table IV: Over-Correction Diagnosis (Holm-Bonferroni Corrected p-values)\n")
        out.append("| Model | Operator / System | ΔMRR (Initially Correct) | p_corr | ΔMRR (Initially Failed) | p_corr |")
        out.append("|---|---|---|---|---|---|")
        
        tex_out.append("\\begin{table*}[t]\n\\caption{Over-Correction Diagnosis across Initially Correct vs Failed Queries (Holm-Corrected)}\n\\label{tab:overcorrection}\n\\centering\n\\begin{tabular}{llcccc}\n\\hline\nModel & Operator & $\\Delta$MRR (Correct) & $p_{corr}$ & $\\Delta$MRR (Failed) & $p_{corr}$ \\\\\n\\hline")
        
        for m, mres in over.items():
            if not isinstance(mres, dict) or "operators" not in mres: continue
            ops = mres.get("operators", {})
            for op, opres in ops.items():
                dc = opres.get("mean_delta_correct", 0.0)
                pc = opres.get("wilcoxon_p_correct_holm", 1.0)
                di = opres.get("mean_delta_incorrect", 0.0)
                pi = opres.get("wilcoxon_p_incorrect_holm", 1.0)
                sig_c = "**" if pc < 0.05 else ""
                sig_i = "**" if pi < 0.05 else ""
                out.append(f"| **{m}** | {op} | {dc:+.4f} {sig_c} | {pc:.4f} | {di:+.4f} {sig_i} | {pi:.4f} |")
                tex_out.append(f"{m} & {op} & {dc:+.4f} & {pc:.4f} & {di:+.4f} & {pi:.4f} \\\\")
                
        out.append("\n")
        tex_out.append("\\hline\n\\end{tabular}\n\\end{table*}\n\n")
    except Exception as e:
        out.append(f"*(Could not load overcorrection: {e})*\n")

    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(out))
    print(f"Markdown tables saved to {OUT_MD}")

    OUT_TEX.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_TEX, "w", encoding="utf-8") as f:
        f.write("\n".join(tex_out))
    print(f"LaTeX tables saved to {OUT_TEX}")

if __name__ == "__main__":
    main()
