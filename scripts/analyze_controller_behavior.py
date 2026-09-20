import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt

def get_cmi_band(cmi):
    if cmi < 0.2: return "low (<0.2)"
    elif cmi < 0.4: return "medium (0.2-0.4)"
    elif cmi < 0.6: return "high (0.4-0.6)"
    else: return "very high (>0.6)"

def get_conf_band(conf):
    if conf < 0.2: return "low (<0.2)"
    elif conf < 0.4: return "medium (0.2-0.4)"
    elif conf < 0.6: return "high (0.4-0.6)"
    elif conf < 0.8: return "very high (0.6-0.8)"
    else: return "extreme (>0.8)"

def main():
    root = Path(__file__).resolve().parents[1]
    
    # Read trajectories
    trajectories = []
    with open(root / "data" / "logs" / "trajectories_v3.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                trajectories.append(json.loads(line))
                
    # Read v2 results
    with open(root / "results" / "logs" / "setu_v2_per_query_v3.json", "r", encoding="utf-8") as f:
        v2_logs = json.load(f)
        
    # Process queries
    query_data = {}
    for row in trajectories:
        qid = row["query_id"]
        if qid not in query_data:
            query_data[qid] = {
                "actions": [],
                "cmi": float(row["state"]["cmi"]),
                "conf_0": float(row["state"]["confidence"]),
                "stop_reason": None,
                "steps": 0
            }
        query_data[qid]["actions"].append(row["action"])
        
    for row in v2_logs:
        qid = row["query_id"]
        if qid in query_data:
            query_data[qid]["stop_reason"] = row.get("stop_reason")
            # Mean steps (excluding STOP) - derived from actions list
            query_data[qid]["steps"] = len([a for a in query_data[qid]["actions"] if a != "STOP"])

    # 1. Unique action sequences
    sequences = ["->".join(d["actions"]) for d in query_data.values()]
    seq_counts = pd.Series(sequences).value_counts().to_dict()
    
    # 2. % queries where STOP was first-choice
    explicit_stop_total = sum(1 for d in query_data.values() if d["actions"][-1] == "STOP" and d["stop_reason"] == "explicit_stop")
    perc_explicit_stop = (explicit_stop_total / len(query_data)) * 100
    
    # 3. Termination by CMI band
    bands_term = {}
    for d in query_data.values():
        band = get_cmi_band(d["cmi"])
        reason = d["stop_reason"]
        if band not in bands_term:
            bands_term[band] = {"explicit_stop": 0, "repeat_action_forced": 0, "max_steps_reached": 0}
        if reason in bands_term[band]:
            bands_term[band][reason] += 1
        else:
            bands_term[band][reason] = 1 # Fallback
            
    # 4. First-action distribution
    first_actions = [d["actions"][0] for d in query_data.values() if d["actions"]]
    first_action_dist = pd.Series(first_actions).value_counts().to_dict()
    
    # 5. Action distribution by CMI band
    cmi_action_counts = {}
    cmi_bands = ["low (<0.2)", "medium (0.2-0.4)", "high (0.4-0.6)", "very high (>0.6)"]
    for b in cmi_bands:
        cmi_action_counts[b] = {"LAG": 0, "CAEP": 0, "LQP": 0, "STOP": 0}
        
    for d in query_data.values():
        band = get_cmi_band(d["cmi"])
        action = d["actions"][0]
        cmi_action_counts[band][action] += 1
        
    # 6. Mean steps by CMI band
    cmi_steps = {b: [] for b in cmi_bands}
    for d in query_data.values():
        cmi_steps[get_cmi_band(d["cmi"])].append(d["steps"])
    mean_steps_cmi = {b: float(np.mean(vals)) if vals else 0.0 for b, vals in cmi_steps.items()}
    
    # 7. Mean steps by confidence band
    conf_bands = ["low (<0.2)", "medium (0.2-0.4)", "high (0.4-0.6)", "very high (0.6-0.8)", "extreme (>0.8)"]
    conf_steps = {b: [] for b in conf_bands}
    for d in query_data.values():
        conf_steps[get_conf_band(d["conf_0"])].append(d["steps"])
    mean_steps_conf = {b: float(np.mean(vals)) if vals else 0.0 for b, vals in conf_steps.items()}
    
    # 8. Chi-square test (First action vs CMI band)
    contingency = []
    actions_order = ["LAG", "CAEP", "LQP", "STOP"]
    bands_with_data = []
    for b in cmi_bands:
        row = [cmi_action_counts[b][a] for a in actions_order]
        # Only include bands that have >0 queries to avoid math domain errors
        if sum(row) > 0:
            contingency.append(row)
            bands_with_data.append(b)
            
    if len(contingency) > 1:
        chi2, p, dof, ex = chi2_contingency(contingency)
    else:
        chi2, p = 0.0, 1.0

    output = {
        "n_queries": len(query_data),
        "unique_sequences": seq_counts,
        "first_action_distribution": first_action_dist,
        "explicit_stop_percentage": perc_explicit_stop,
        "termination_by_cmi_band": bands_term,
        "first_action_by_cmi_band": cmi_action_counts,
        "mean_steps_by_cmi": mean_steps_cmi,
        "mean_steps_by_confidence": mean_steps_conf,
        "chi_square_test": {
            "chi2_statistic": float(chi2),
            "p_value": float(p),
            "significant_at_05": bool(p < 0.05)
        }
    }
    
    out_dir = root / "results" / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "controller_behavior_final.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4)
        
    # Plots
    fig_dir = root / "results" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(14, 6))
    
    # Plot 1: Sequence Distribution
    plt.subplot(1, 2, 1)
    seqs = list(seq_counts.keys())
    counts = list(seq_counts.values())
    plt.bar(seqs, counts, color='skyblue')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Frequency')
    plt.title('Action Sequence Distribution')
    
    # Plot 2: First Action by CMI Band
    plt.subplot(1, 2, 2)
    x = np.arange(len(bands_with_data))
    width = 0.2
    
    lag_counts = [cmi_action_counts[b]["LAG"] for b in bands_with_data]
    caep_counts = [cmi_action_counts[b]["CAEP"] for b in bands_with_data]
    lqp_counts = [cmi_action_counts[b]["LQP"] for b in bands_with_data]
    stop_counts = [cmi_action_counts[b]["STOP"] for b in bands_with_data]
    
    plt.bar(x - 1.5*width, lag_counts, width, label='LAG')
    plt.bar(x - 0.5*width, caep_counts, width, label='CAEP')
    plt.bar(x + 0.5*width, lqp_counts, width, label='LQP')
    plt.bar(x + 1.5*width, stop_counts, width, label='STOP')
    
    plt.xlabel('CMI Band')
    plt.ylabel('Count')
    plt.title('First Action by CMI Band')
    plt.xticks(x, bands_with_data, rotation=45, ha='right')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(fig_dir / "controller_behavior.png")
    
if __name__ == "__main__":
    main()
