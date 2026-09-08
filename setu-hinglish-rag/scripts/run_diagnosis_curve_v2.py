import json
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import scipy.stats as stats
import faiss
import matplotlib.pyplot as plt
from ranx import Qrels, Run, evaluate

sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path("c:/Users/sumas/Downloads/setu-hinglish-rag-skeleton/setu-hinglish-rag")
sys.path.append(str(ROOT))

import setu.config as config
from setu.diagnosis.cmi import cmi
from setu.diagnosis.lid_entropy import lid_entropy

DATA_DIR = ROOT / "data"
LOG_DIR = ROOT / "results" / "logs"
TABLE_DIR = ROOT / "results" / "tables"
FIG_DIR = ROOT / "results" / "figures"

LOG_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# 1. Load Scaled Data
corpus_v2_file = DATA_DIR / "processed" / "corpus_chunks_v2.jsonl"
queries_v3_file = DATA_DIR / "processed" / "queries_v3_final.json"

chunks_v2 = [json.loads(l) for l in open(corpus_v2_file, encoding='utf-8') if l.strip()]
queries_v3 = json.load(open(queries_v3_file, encoding='utf-8'))

doc_ids_v2 = [c["chunk_id"] for c in chunks_v2]
doc_texts_v2 = [c["text"] for c in chunks_v2]
query_ids_v3 = [q["query_id"] for q in queries_v3]
query_texts_v3 = [q["text"] for q in queries_v3]

print(f"Loaded {len(chunks_v2)} corpus chunks (v2).")
print(f"Loaded {len(queries_v3)} queries (v3).")

# 2. Compute CMI and LID-entropy for each query
def cmi_band(score, bands):
    for lo, hi, name in bands:
        if lo <= score < hi:
            return name
    return bands[-1][2]

query_diagnostics = {}
query_bands = {}

cmi_scores = []
lid_scores = []

for q in queries_v3:
    qid = q["query_id"]
    text = q["text"]
    c = cmi(text)
    lid = lid_entropy(text)
    band = cmi_band(c, config.CMI_BANDS)
    
    query_diagnostics[qid] = {
        "text": text,
        "cmi": c,
        "lid_entropy": lid,
        "cmi_band": band,
        "relevant_doc_ids": q["relevant_doc_ids"]
    }
    query_bands[qid] = band
    cmi_scores.append(c)
    lid_scores.append(lid)

print(f"Computed CMI & LID-entropy for all {len(queries_v3)} queries.")
print(f"CMI Distribution: Mean={np.mean(cmi_scores):.4f} (±{np.std(cmi_scores):.4f}), Median={np.median(cmi_scores):.4f}, Range=[{np.min(cmi_scores):.4f}, {np.max(cmi_scores):.4f}]")
print(f"LID-entropy Distribution: Mean={np.mean(lid_scores):.4f} (±{np.std(lid_scores):.4f}), Median={np.median(lid_scores):.4f}, Range=[{np.min(lid_scores):.4f}, {np.max(lid_scores):.4f}]")

# Count queries per CMI band
BAND_ORDER = [b[2] for b in config.CMI_BANDS]
band_counts = {b: sum(1 for band in query_bands.values() if band == b) for b in BAND_ORDER}
print("Queries per CMI band:", band_counts)

# 3. Load Embeddings & Run FAISS Retrieval
MODELS = ["bge_m3", "indic_sbert", "me5_large"]
METRICS = ["ndcg@5", "ndcg@10", "mrr", "recall@5", "recall@10", "hit_rate@1"]

qrels_dict = {q["query_id"]: {d: 1 for d in q["relevant_doc_ids"]} for q in queries_v3}
qrels = Qrels(qrels_dict)

per_query_results = {}
per_band_summary = {}

for model_key in MODELS:
    doc_emb_file = DATA_DIR / "embeddings" / f"doc_emb_{model_key}_v2.npy"
    query_emb_file = DATA_DIR / "embeddings" / f"query_emb_{model_key}_v3.npy"
    
    d_emb = np.load(doc_emb_file)
    q_emb = np.load(query_emb_file)
    
    index = faiss.IndexFlatIP(d_emb.shape[1])
    index.add(d_emb)
    scores, indices = index.search(q_emb, 380)
    
    run_dict = {}
    for i, qid in enumerate(query_ids_v3):
        ranked_doc_ids = [doc_ids_v2[idx] for idx in indices[i][:10]]
        ranked_scores = [float(s) for s in scores[i][:10]]
        run_dict[qid] = {d: s for d, s in zip(ranked_doc_ids, ranked_scores)}
        
    run = Run(run_dict)
    
    # Overall evaluation
    overall_metrics = evaluate(qrels, run, METRICS)
    print(f"\nOverall Metrics ({model_key}):", overall_metrics)
    
    # Per-query metrics
    per_q_eval = evaluate(qrels, run, METRICS, return_mean=False)
    per_query_results[model_key] = {
        qid: {m: float(per_q_eval[m][i]) for m in METRICS}
        for i, qid in enumerate(query_ids_v3)
    }
    
    # Per-band aggregation
    band_metrics = {}
    for band in BAND_ORDER:
        q_in_band = [qid for qid, b in query_bands.items() if b == band]
        if not q_in_band:
            continue
        band_metrics[band] = {
            "count": len(q_in_band),
            "recall@5": float(np.mean([per_query_results[model_key][qid]["recall@5"] for qid in q_in_band])),
            "recall@10": float(np.mean([per_query_results[model_key][qid]["recall@10"] for qid in q_in_band])),
            "mrr": float(np.mean([per_query_results[model_key][qid]["mrr"] for qid in q_in_band])),
            "ndcg@5": float(np.mean([per_query_results[model_key][qid]["ndcg@5"] for qid in q_in_band])),
            "ndcg@10": float(np.mean([per_query_results[model_key][qid]["ndcg@10"] for qid in q_in_band])),
            "hit_rate@1": float(np.mean([per_query_results[model_key][qid]["hit_rate@1"] for qid in q_in_band]))
        }
    per_band_summary[model_key] = band_metrics

# 4. Save results to results/tables/degradation_curve_v2.json and results/logs/
out_table_file = TABLE_DIR / "degradation_curve_v2.json"
table_data = {
    "cmi_bands": config.CMI_BANDS,
    "band_counts": band_counts,
    "per_band_summary": per_band_summary,
    "per_query_diagnostics": query_diagnostics
}
with open(out_table_file, "w", encoding="utf-8") as f:
    json.dump(table_data, f, indent=2)
print(f"\nSaved degradation curve data to: {out_table_file}")

# 5. Plot and save degradation curve figure (results/figures/degradation_curve_v2.png)
model_display_names = {
    "bge_m3": "BGE-M3",
    "indic_sbert": "Indic-SBERT",
    "me5_large": "mE5-Large"
}
colors = {
    "bge_m3": "#1f77b4",
    "indic_sbert": "#d62728",
    "me5_large": "#2ca02c"
}

fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))
metrics_to_plot = [
    ("recall@5", "Recall@5", axes[0]),
    ("mrr", "MRR", axes[1]),
    ("ndcg@5", "nDCG@5", axes[2])
]

x_labels = [f"{b.replace('_', ' ').title()}\n(n={band_counts[b]})" for b in BAND_ORDER]

for m_key, m_name, ax in metrics_to_plot:
    for model_key in MODELS:
        y_vals = [per_band_summary[model_key][b][m_key] for b in BAND_ORDER]
        ax.plot(x_labels, y_vals, marker="o", linewidth=2.2, markersize=7,
                color=colors[model_key], label=model_display_names[model_key])
        
        # Annotate points
        for xi, yi in enumerate(y_vals):
            ax.annotate(f"{yi:.3f}", (xi, yi), textcoords="offset points", xytext=(0, 7),
                        ha='center', fontsize=8.5, fontweight='semibold')
            
    ax.set_title(f"{m_name} vs. Code-Mixing Index (CMI)", fontsize=13, fontweight='bold')
    ax.set_xlabel("CMI Band", fontsize=11)
    ax.set_ylabel(m_name, fontsize=11)
    ax.set_ylim(0.40, 1.05)
    ax.grid(True, linestyle="--", alpha=0.5)
    if ax == axes[0]:
        ax.legend(fontsize=10, loc="lower left")

plt.suptitle("SETU Scaled Degradation Diagnosis Curve: 380 Chunks × 314 Queries", fontsize=15, y=1.02)
plt.tight_layout()

out_fig = FIG_DIR / "degradation_curve_v2.png"
plt.savefig(out_fig, dpi=300, bbox_inches="tight")
plt.close()
print(f"Saved scaled degradation curve plot to: {out_fig}")

# 6. Statistical Comparison: Slope & Correlation Analysis (Pilot vs Scaled)
print("\n" + "="*95)
print("PER-CMI-BAND RETRIEVAL DEGRADATION TABLE (380-CHUNK CORPUS, 314 QUERIES)")
print("="*95)
for model_key in MODELS:
    print(f"\nModel: {model_display_names[model_key]}")
    print(f"{'Band':<15} {'Count':<8} {'Recall@5':<12} {'Recall@10':<12} {'MRR':<12} {'nDCG@5':<12} {'nDCG@10':<12}")
    print("-" * 83)
    for band in BAND_ORDER:
        b_data = per_band_summary[model_key][band]
        print(f"{band:<15} {b_data['count']:<8} {b_data['recall@5']:<12.4f} {b_data['recall@10']:<12.4f} {b_data['mrr']:<12.4f} {b_data['ndcg@5']:<12.4f} {b_data['ndcg@10']:<12.4f}")

# Compute regression slope and Spearman correlation (CMI vs Recall@5, MRR, nDCG@5)
print("\n" + "="*95)
print("STATISTICAL ANALYSIS: CMI DEGRADATION SLOPE & CORRELATION (SCALED vs PILOT)")
print("="*95)
print(f"{'Model':<15} | {'Metric':<10} | {'Scaled Slope (per 1.0 CMI)':<28} | {'Scaled Spearman rho':<22} | {'p-value':<12}")
print("-" * 95)

cmi_array = np.array([query_diagnostics[qid]["cmi"] for qid in query_ids_v3])
lid_array = np.array([query_diagnostics[qid]["lid_entropy"] for qid in query_ids_v3])

for model_key in MODELS:
    for m in ["recall@5", "mrr", "ndcg@5"]:
        scores = np.array([per_query_results[model_key][qid][m] for qid in query_ids_v3])
        slope, intercept, r_val, p_val, std_err = stats.linregress(cmi_array, scores)
        rho, p_rho = stats.spearmanr(cmi_array, scores)
        print(f"{model_display_names[model_key]:<15} | {m:<10} | {slope:<28.4f} | {rho:<22.4f} | {p_rho:<12.4e}")
    print("-" * 95)

out_per_query = LOG_DIR / "per_query_metrics_v2.json"
with open(out_per_query, "w", encoding="utf-8") as f:
    json.dump(per_query_results, f, indent=2)
print(f"Saved scaled per-query metrics to {out_per_query}")

