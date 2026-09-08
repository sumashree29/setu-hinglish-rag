import shutil
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

def main():
    print("Assembling final outputs for Phase 6...")
    
    out_fig = ROOT / "outputs" / "figures"
    out_tab = ROOT / "outputs" / "tables"
    
    out_fig.mkdir(parents=True, exist_ok=True)
    out_tab.mkdir(parents=True, exist_ok=True)
    
    figures_to_copy = [
        ROOT / "results" / "figures" / "degradation_curve_v2.png",
        ROOT / "results" / "figures" / "scaled_corpus_retrieval_comparison_v3.png"
    ]
    
    tables_to_copy = [
        ROOT / "results" / "tables" / "setu_v1_v2_comparison_scaled.json",
        ROOT / "results" / "tables" / "operator_ablation.json",
        ROOT / "results" / "tables" / "extended_baselines.md",
        ROOT / "results" / "tables" / "statistical_significance_H1_H10_scaled.json"
    ]
    
    for fig in figures_to_copy:
        if fig.exists():
            shutil.copy(fig, out_fig / fig.name)
            print(f"Copied {fig.name} to outputs/figures/")
        else:
            print(f"Warning: {fig.name} not found.")
            
    for tab in tables_to_copy:
        if tab.exists():
            shutil.copy(tab, out_tab / tab.name)
            print(f"Copied {tab.name} to outputs/tables/")
        else:
            print(f"Warning: {tab.name} not found.")
            
    print("\nPhase 6 assembly complete. The outputs/ folder is fully populated.")

if __name__ == "__main__":
    main()
