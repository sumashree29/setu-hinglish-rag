import json
import sys
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results" / "tables"
FIG_DIR = ROOT / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

def generate_figure_1():
    """
    Figure 1: MRR degradation curve against CMI (line plot with confidence intervals).
    """
    print("Generating Figure 1 (MRR vs CMI)...")
    try:
        per_query = json.load(open(ROOT / "results" / "logs" / "per_query_metrics_v2.json", encoding="utf-8"))
        queries = json.load(open(ROOT / "data" / "processed" / "queries_v3_final.json", encoding="utf-8"))
        
        sys.path.append(str(ROOT))
        from setu.diagnosis.cmi import cmi
        
        cmis = []
        mrrs = []
        for q in queries:
            qid = q["query_id"]
            if qid in per_query["bge_m3"]:
                cmis.append(cmi(q["text"]))
                mrrs.append(per_query["bge_m3"][qid]["mrr"])
                
        # Group by CMI bins
        bins = np.linspace(0, 1.0, 11)
        bin_indices = np.digitize(cmis, bins)
        
        bin_centers = []
        mean_mrrs = []
        err_mrrs = []
        
        for i in range(1, len(bins)):
            b_mrrs = [mrrs[j] for j in range(len(mrrs)) if bin_indices[j] == i]
            if b_mrrs:
                bin_centers.append((bins[i-1] + bins[i]) / 2)
                mean_mrrs.append(np.mean(b_mrrs))
                err_mrrs.append(1.96 * np.std(b_mrrs) / np.sqrt(len(b_mrrs)))
                
        plt.figure(figsize=(8, 5))
        plt.errorbar(bin_centers, mean_mrrs, yerr=err_mrrs, fmt='-o', capsize=5, label='BGE-M3 RAW')
        plt.xlabel("Code-Mixing Index (CMI)")
        plt.ylabel("Mean Reciprocal Rank (MRR)")
        plt.title("Figure 1: MRR Degradation against CMI")
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.savefig(FIG_DIR / "fig1_mrr_vs_cmi.png", dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"Error generating Figure 1: {e}")

def generate_table_1():
    """
    Table 1: H1-H10 hypotheses summary table (LaTeX format).
    """
    print("Generating Table 1 (LaTeX Hypotheses Summary)...")
    try:
        stats = json.load(open(RESULTS_DIR / "statistical_significance_H1_H10_scaled.json", encoding="utf-8"))
        
        latex = [
            "\\begin{table*}[t]",
            "\\centering",
            "\\small",
            "\\begin{tabular}{p{0.05\\linewidth} p{0.45\\linewidth} p{0.15\\linewidth} p{0.1\\linewidth} p{0.15\\linewidth}}",
            "\\toprule",
            "\\textbf{ID} & \\textbf{Hypothesis} & \\textbf{Statistic} & \\textbf{p-value} & \\textbf{Verdict} \\\\",
            "\\midrule"
        ]
        
        for h_id in ["H1", "H2", "H3", "H4", "H5", "H6", "H7", "H8", "H9", "H10"]:
            if h_id in stats:
                data = stats[h_id]
                hyp = data["hypothesis"].split(":")[0] if ":" in data["hypothesis"] else h_id
                
                # Format stat
                if data["statistic"] is None or np.isnan(data["statistic"]):
                    stat_str = "---"
                    p_str = "---"
                else:
                    stat_str = f"{data['statistic']:.3f}"
                    p_val = data.get("p_value_corrected", data["p_value"])
                    if p_val is None or np.isnan(p_val):
                        p_str = "---"
                    else:
                        p_str = f"{p_val:.3e}" if p_val < 0.001 else f"{p_val:.3f}"
                        
                verdict = data["verdict"].replace("_", "\\_").title()
                
                latex.append(f"{h_id} & {hyp} & {stat_str} & {p_str} & {verdict} \\\\")
                
        latex.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\caption{Summary of Statistical Hypotheses Testing (Holm-corrected $p$-values)}",
            "\\label{tab:hypotheses}",
            "\\end{table*}"
        ])
        
        with open(FIG_DIR / "table1_hypotheses.tex", "w", encoding="utf-8") as f:
            f.write("\n".join(latex))
            
    except Exception as e:
        print(f"Error generating Table 1: {e}")

def generate_figure_2():
    """
    Figure 2: The 2x2 outcome-by-termination cross-tab from Task 9 as a stacked bar chart.
    """
    print("Generating Figure 2 (Stacked Bar Chart for Termination Cause)...")
    try:
        data = json.load(open(RESULTS_DIR / "error_analysis_v2.json", encoding="utf-8"))
        crosstab = data.get("outcome_by_termination_cause_crosstab", {})
        
        labels = []
        success = []
        failure = []
        
        for reason, outcomes in crosstab.items():
            labels.append(reason.replace("_", " ").title())
            success.append(outcomes.get("success", 0))
            failure.append(outcomes.get("failure", 0))
            
        x = np.arange(len(labels))
        width = 0.5
        
        plt.figure(figsize=(8, 6))
        plt.bar(x, success, width, label='Success (MRR=1.0)', color='#2ca02c')
        plt.bar(x, failure, width, bottom=success, label='Failure (MRR<1.0)', color='#d62728')
        
        plt.ylabel('Number of Queries')
        plt.title('Figure 2: SETU v2 Outcomes by Termination Cause')
        plt.xticks(x, labels, rotation=15)
        plt.legend()
        
        # Add labels on bars
        for i in range(len(labels)):
            total = success[i] + failure[i]
            if total > 0:
                plt.text(x[i], total + max(success+failure)*0.01, str(total), ha='center')
                
        plt.tight_layout()
        plt.savefig(FIG_DIR / "fig2_outcome_by_termination.png", dpi=300)
        plt.close()
    except Exception as e:
        print(f"Error generating Figure 2: {e}")

if __name__ == "__main__":
    generate_figure_1()
    generate_table_1()
    generate_figure_2()
    print(f"Saved all figures and tables to {FIG_DIR}")
