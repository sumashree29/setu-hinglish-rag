import json
import numpy as np
from pathlib import Path
from scipy.stats import chi2_contingency
from collections import Counter
import math

ROOT = Path(__file__).resolve().parents[1]

def entropy(labels):
    counts = Counter(labels)
    total = len(labels)
    ent = 0.0
    for count in counts.values():
        p = count / total
        ent -= p * math.log2(p)
    return ent

def main():
    per_query = []
    with open(ROOT / "results" / "canonical" / "per_query_results.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            per_query.append(json.loads(line))
            
    queries = json.load(open(ROOT / "data" / "processed" / "queries_v3_final.json", "r", encoding="utf-8"))
    cmi_dict = {q["query_id"]: q.get("cmi", 0.0) for q in queries}
            
    initial_actions = []
    sequences = []
    stop_reasons = []
    cmis = []
    
    for q in per_query:
        seq = tuple(q["v2_action_sequence"])
        sequences.append(seq)
        initial_actions.append(seq[0] if seq else "STOP")
        stop_reasons.append(q["v2_stop_reason"])
        cmis.append(cmi_dict.get(q["query_id"], 0.0))
        
    init_dist = dict(Counter(initial_actions))
    seq_dist = dict(Counter([str(s) for s in sequences]))
    seq_ent = entropy(sequences)
    stop_dist = dict(Counter(stop_reasons))
    
    # CMI Band vs Initial Action Chi-square
    # Let's define CMI bands: low (< 10), med (10-25), high (>25)
    cmi_bands = []
    for c in cmis:
        if c < 10: cmi_bands.append("low")
        elif c < 25: cmi_bands.append("med")
        else: cmi_bands.append("high")
        
    contingency = {}
    for action in set(initial_actions):
        contingency[action] = {"low": 0, "med": 0, "high": 0}
        
    for act, bnd in zip(initial_actions, cmi_bands):
        contingency[act][bnd] += 1
        
    matrix = []
    actions_in_matrix = []
    for act, row in contingency.items():
        if sum(row.values()) > 0:
            matrix.append([row["low"], row["med"], row["high"]])
            actions_in_matrix.append(act)
            
    chi2 = None
    p_val = None
    dof = None
    if len(matrix) > 1: # need at least 2 categories for independence test
        try:
            chi2, p_val, dof, expected = chi2_contingency(matrix)
        except Exception:
            pass

    out = {
        "distributions": {
            "initial_action": init_dist,
            "full_sequence": seq_dist,
            "stop_reasons": stop_dist
        },
        "metrics": {
            "unique_sequence_count": len(seq_dist),
            "sequence_entropy": seq_ent,
            "explicit_stop_rate": stop_dist.get("explicit_stop", 0) / len(per_query),
            "forced_repeat_stop_rate": stop_dist.get("repeat_action_forced", 0) / len(per_query)
        },
        "chi_square_cmi_vs_initial_action": {
            "contingency_table": contingency,
            "chi2_statistic": float(chi2) if chi2 is not None else None,
            "p_value": float(p_val) if p_val is not None else None,
            "dof": int(dof) if dof is not None else None,
            "interpretation": "Controller action choice is completely independent of CMI complexity" if (p_val is not None and p_val > 0.05) else "Controller action correlates with CMI"
        }
    }
    
    with open(ROOT / "results" / "canonical" / "controller_results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=4)
        
    print(f"Controller analysis saved. Sequence entropy: {seq_ent:.2f}. Chi2 p: {p_val}")

if __name__ == "__main__":
    main()
