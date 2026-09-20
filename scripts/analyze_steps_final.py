import json
import numpy as np
from pathlib import Path
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]

def main():
    per_query = []
    with open(ROOT / "results" / "canonical" / "per_query_results.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            per_query.append(json.loads(line))
            
    v1_steps = []
    v2_steps = []
    
    for q in per_query:
        v1_steps.append(q["v1_steps"])
        v2_actual = len([a for a in q["v2_action_sequence"] if a != "STOP"])
        v2_steps.append(v2_actual)
        
    v1_steps = np.array(v1_steps)
    v2_steps = np.array(v2_steps)
    
    diff = v1_steps - v2_steps
    # reduction % from v1
    red_pct = (diff / np.where(v1_steps > 0, v1_steps, 1)) * 100
    
    stat, pval = wilcoxon(v1_steps, v2_steps)
    
    results = {
        "definition": "One step = one executed non-STOP operator action.",
        "v1": {
            "mean": float(np.mean(v1_steps)),
            "median": float(np.median(v1_steps)),
            "std": float(np.std(v1_steps)),
            "iqr": float(np.percentile(v1_steps, 75) - np.percentile(v1_steps, 25)),
            "p95": float(np.percentile(v1_steps, 95))
        },
        "v2": {
            "mean": float(np.mean(v2_steps)),
            "median": float(np.median(v2_steps)),
            "std": float(np.std(v2_steps)),
            "iqr": float(np.percentile(v2_steps, 75) - np.percentile(v2_steps, 25)),
            "p95": float(np.percentile(v2_steps, 95))
        },
        "comparison": {
            "mean_reduction_absolute": float(np.mean(diff)),
            "mean_reduction_percentage": float(np.mean(red_pct)),
            "wilcoxon_statistic": float(stat),
            "wilcoxon_p": float(pval)
        }
    }
    
    with open(ROOT / "results" / "canonical" / "step_analysis.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
        
    print(f"Step analysis complete. Saved to results/canonical/step_analysis.json")
    print(f"v1 mean steps: {results['v1']['mean']:.2f}")
    print(f"v2 mean steps: {results['v2']['mean']:.2f}")
    print(f"Reduction: {results['comparison']['mean_reduction_percentage']:.2f}% (p={pval:.4e})")

if __name__ == "__main__":
    main()
