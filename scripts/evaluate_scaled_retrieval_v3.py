import json
import re
import sys
import time
from pathlib import Path
import numpy as np
import scipy.stats as stats
import faiss
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from ranx import Qrels, Run, evaluate

sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path("c:/Users/sumas/Downloads/setu-hinglish-rag-skeleton/setu-hinglish-rag")
DATA_DIR = ROOT / "data"
EMB_DIR = DATA_DIR / "embeddings"
TABLE_DIR = ROOT / "results" / "tables"
FIG_DIR = ROOT / "results" / "figures"
LOG_DIR = ROOT / "results" / "logs"

EMB_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# 1. Load data
corpus_v2_file = DATA_DIR / "processed" / "corpus_chunks_v2.jsonl"
queries_v3_file = DATA_DIR / "processed" / "queries_v3_final.json"

chunks_v2 = [json.loads(l) for l in open(corpus_v2_file, encoding='utf-8') if l.strip()]
chunks_v2_map = {c["chunk_id"]: c for c in chunks_v2}
queries_v3 = json.load(open(queries_v3_file, encoding='utf-8'))

print(f"Loaded {len(chunks_v2)} corpus chunks (v2).")
print(f"Loaded {len(queries_v3)} queries (v3).")

doc_ids_v2 = [c["chunk_id"] for c in chunks_v2]
doc_texts_v2 = [c["text"] for c in chunks_v2]

query_ids_v3 = [q["query_id"] for q in queries_v3]
query_texts_v3 = [q["text"] for q in queries_v3]

pilot_q_ids = set(q["query_id"] for q in queries_v3[:75]) # Q01-Q75
auto_q_ids = set(q["query_id"] for q in queries_v3[75:])  # Q076-Q314

# Pilot corpus (20 chunks: D01-D20)
chunks_pilot = chunks_v2[:20]
doc_ids_pilot = [c["chunk_id"] for c in chunks_pilot]
doc_texts_pilot = [c["text"] for c in chunks_pilot]

MODELS = {
    "bge_m3": ("BAAI/bge-m3", False),
    "indic_sbert": ("l3cube-pune/indic-sentence-similarity-sbert", False),
    "me5_large": ("intfloat/multilingual-e5-large", True),
}

# 2. Embedding Generation & Caching (Re-generate queries, reuse docs if identical)
embeddings_v2 = {}

for model_key, (model_name, needs_prefix) in MODELS.items():
    print(f"\n==================================================")
    print(f"Embedding with {model_key} ({model_name})")
    print(f"==================================================")
    
    t0 = time.time()
    model = SentenceTransformer(model_name)
    
    # Doc embeddings v2 (380 chunks)
    doc_emb_path = EMB_DIR / f"doc_emb_{model_key}_v2.npy"
    if doc_emb_path.exists():
        print(f"Loading existing doc embeddings from {doc_emb_path}...")
        d_emb_v2 = np.load(doc_emb_path)
    else:
        d_input_v2 = [f"passage: {t}" for t in doc_texts_v2] if needs_prefix else doc_texts_v2
        print(f"Encoding {len(d_input_v2)} corpus chunks...")
        d_emb_v2 = model.encode(d_input_v2, show_progress_bar=True, convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(d_emb_v2)
        np.save(doc_emb_path, d_emb_v2)
        np.save(LOG_DIR / f"doc_emb_{model_key}_v2.npy", d_emb_v2)
    
    # Query embeddings v3 (314 queries) - always regenerate since queries were paraphrased
    q_input_v3 = [f"query: {t}" for t in query_texts_v3] if needs_prefix else query_texts_v3
    print(f"Encoding {len(q_input_v3)} paraphrased queries...")
    q_emb_v3 = model.encode(q_input_v3, show_progress_bar=True, convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(q_emb_v3)
    
    query_emb_path = EMB_DIR / f"query_emb_{model_key}_v3.npy"
    np.save(query_emb_path, q_emb_v3)
    np.save(LOG_DIR / f"query_emb_{model_key}_v3.npy", q_emb_v3)
    
    print(f"Saved query embeddings to {query_emb_path} shape: {q_emb_v3.shape}")
    print(f"Elapsed: {time.time() - t0:.2f}s")
    
    embeddings_v2[model_key] = {
        "doc_emb_v2": d_emb_v2,
        "query_emb_v3": q_emb_v3,
        "doc_emb_pilot": d_emb_v2[:20],
        "query_emb_pilot": q_emb_v3[:75]
    }

# 3. Build Qrels
qrels_all_dict = {q["query_id"]: {d: 1 for d in q["relevant_doc_ids"]} for q in queries_v3}
qrels_all = Qrels(qrels_all_dict)

qrels_pilot_dict = {q["query_id"]: {d: 1 for d in q["relevant_doc_ids"]} for q in queries_v3[:75]}
qrels_pilot = Qrels(qrels_pilot_dict)

qrels_auto_dict = {q["query_id"]: {d: 1 for d in q["relevant_doc_ids"]} for q in queries_v3[75:]}
qrels_auto = Qrels(qrels_auto_dict)

METRICS = ["recall@5", "recall@10", "mrr", "ndcg@10", "hit_rate@1"]

results_summary = {}
mrr_per_query = {m: [] for m in MODELS}

for model_key in MODELS.keys():
    d_emb_v2 = embeddings_v2[model_key]["doc_emb_v2"]
    q_emb_v3 = embeddings_v2[model_key]["query_emb_v3"]
    
    d_emb_pilot = embeddings_v2[model_key]["doc_emb_pilot"]
    q_emb_pilot = embeddings_v2[model_key]["query_emb_pilot"]
    
    # --- Scaled Corpus (314 queries against 380 chunks) ---
    index_v2 = faiss.IndexFlatIP(d_emb_v2.shape[1])
    index_v2.add(d_emb_v2)
    scores_v2, indices_v2 = index_v2.search(q_emb_v3, 380)
    
    run_all_dict = {}
    for i, q_id in enumerate(query_ids_v3):
        run_all_dict[q_id] = {
            doc_ids_v2[idx]: float(scores_v2[i, rank])
            for rank, idx in enumerate(indices_v2[i][:10])
        }
        
        # Calculate full RR for correlation analysis
        rel_set = set(queries_v3[i]["relevant_doc_ids"])
        rr = 0.0
        for rank_idx in range(len(indices_v2[i])):
            ret_doc = doc_ids_v2[indices_v2[i][rank_idx]]
            if ret_doc in rel_set:
                rr = 1.0 / (rank_idx + 1)
                break
        mrr_per_query[model_key].append(rr)
        
    run_all = Run(run_all_dict)
    
    metrics_all = evaluate(qrels_all, run_all, metrics=METRICS)
    
    run_pilot_on_380_dict = {qid: run_all_dict[qid] for qid in pilot_q_ids}
    run_pilot_on_380 = Run(run_pilot_on_380_dict)
    metrics_pilot_on_380 = evaluate(qrels_pilot, run_pilot_on_380, metrics=METRICS)
    
    run_auto_dict = {qid: run_all_dict[qid] for qid in auto_q_ids}
    run_auto = Run(run_auto_dict)
    metrics_auto = evaluate(qrels_auto, run_auto, metrics=METRICS)
    
    # --- Pilot Baseline (75 pilot queries against 20 pilot chunks) ---
    index_pilot = faiss.IndexFlatIP(d_emb_pilot.shape[1])
    index_pilot.add(d_emb_pilot)
    scores_p, indices_p = index_pilot.search(q_emb_pilot, 10)
    
    run_pilot_20_dict = {}
    for i, q_id in enumerate(query_ids_v3[:75]):
        run_pilot_20_dict[q_id] = {
            doc_ids_pilot[idx]: float(scores_p[i, rank])
            for rank, idx in enumerate(indices_p[i])
        }
    run_pilot_20 = Run(run_pilot_20_dict)
    metrics_pilot_on_20 = evaluate(qrels_pilot, run_pilot_20, metrics=METRICS)
    
    results_summary[model_key] = {
        "overall_314_queries_380_chunks": metrics_all,
        "split_75_pilot_queries_380_chunks": metrics_pilot_on_380,
        "split_239_auto_queries_380_chunks": metrics_auto,
        "baseline_75_pilot_queries_20_chunks": metrics_pilot_on_20
    }

# 4. Lexical Overlap Analysis
STOPWORDS = {
    "a", "an", "the", "in", "on", "at", "to", "for", "of", "with", "by", "from", 
    "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "will", "would", "should", "can", "could", "may", "might", "must",
    "and", "or", "but", "if", "then", "else", "when", "where", "why", "how", "what", "which", "who", "whom",
    "this", "that", "these", "those", "it", "its", "they", "them", "their", "we", "our", "you", "your", "i", "my",
    "kya", "hai", "hain", "hota", "hoti", "hote", "ke", "ki", "ka", "ko", "se", "mein", "par", "bhi", "toh", "aur",
    "ya", "nahi", "kar", "kare", "karna", "karte", "sakte", "sakta", "chahiye", "baare", "jaankari", "kitna", "kitni", "kitne"
}

def compute_overlap(q_text, chunk_text):
    q_tokens = [w.lower() for w in re.findall(r"\b\w+\b", q_text)]
    content_tokens = [w for w in q_tokens if w not in STOPWORDS and len(w) > 1]
    if not content_tokens:
        return 0.0
    chunk_tokens_set = set(w.lower() for w in re.findall(r"\b\w+\b", chunk_text))
    matched_tokens = [w for w in content_tokens if w in chunk_tokens_set]
    return len(matched_tokens) / len(content_tokens)

pilot_overlaps = []
auto_overlaps = []

for idx, q in enumerate(queries_v3):
    rel_docs = q["relevant_doc_ids"]
    target_text = " ".join(chunks_v2_map[d]["text"] for d in rel_docs if d in chunks_v2_map)
    ov = compute_overlap(q["text"], target_text)
    if idx < 75:
        pilot_overlaps.append(ov)
    else:
        auto_overlaps.append(ov)

all_overlaps = pilot_overlaps + auto_overlaps

# Save metrics table
out_table_file = TABLE_DIR / "scaled_corpus_retrieval_v3.json"
with open(out_table_file, "w", encoding="utf-8") as f:
    json.dump(results_summary, f, indent=2)

print(f"\nSaved scaled retrieval metrics table to: {out_table_file}")

# 5. Generate Comparison Chart: Pilot vs Scaled on Pilot Queries
models = list(MODELS.keys())
model_labels = ["BGE-M3", "Indic-SBERT", "mE5-Large"]

fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
metrics_to_plot = [("recall@5", "Recall@5"), ("recall@10", "Recall@10"), ("mrr", "MRR"), ("ndcg@10", "nDCG@10")]

x = np.arange(len(models))
width = 0.35

for ax_idx, (m_key, m_title) in enumerate(metrics_to_plot):
    pilot_20_vals = [results_summary[m]["baseline_75_pilot_queries_20_chunks"][m_key] for m in models]
    scaled_380_vals = [results_summary[m]["split_75_pilot_queries_380_chunks"][m_key] for m in models]
    
    rects1 = axes[ax_idx].bar(x - width/2, pilot_20_vals, width, label='Pilot (20 Chunks)', color='#4C72B0')
    rects2 = axes[ax_idx].bar(x + width/2, scaled_380_vals, width, label='Scaled (380 Chunks)', color='#55A868')
    
    axes[ax_idx].set_title(m_title, fontsize=13, fontweight='bold')
    axes[ax_idx].set_xticks(x)
    axes[ax_idx].set_xticklabels(model_labels, fontsize=11)
    axes[ax_idx].set_ylim(0, 1.05)
    axes[ax_idx].grid(axis='y', linestyle='--', alpha=0.6)
    
    for rect in rects1:
        h = rect.get_height()
        axes[ax_idx].annotate(f'{h:.3f}', xy=(rect.get_x() + rect.get_width() / 2, h),
                              xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
    for rect in rects2:
        h = rect.get_height()
        axes[ax_idx].annotate(f'{h:.3f}', xy=(rect.get_x() + rect.get_width() / 2, h),
                              xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)

axes[0].legend(loc='lower left', fontsize=10)
plt.suptitle("Raw Retrieval Scaling Evaluation: 20 Chunks vs 380 Chunks (75 Pilot Queries)", fontsize=14, y=1.02)
plt.tight_layout()

out_fig_file = FIG_DIR / "scaled_corpus_retrieval_comparison_v3.png"
plt.savefig(out_fig_file, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved comparison figure to: {out_fig_file}")

# 6. Display formatted outputs
print("\n" + "="*80)
print("LEXICAL OVERLAP RESULTS (PILOT VS PARAPHRASED AUTO-GENERATED)")
print("="*80)
print(f"Pilot Queries Overlap (n=75):       Mean = {np.mean(pilot_overlaps):.4f} (±{np.std(pilot_overlaps):.4f}), Median = {np.median(pilot_overlaps):.4f}")
print(f"Auto-Generated Overlap (n=239):     Mean = {np.mean(auto_overlaps):.4f} (±{np.std(auto_overlaps):.4f}), Median = {np.median(auto_overlaps):.4f}")
print(f"Full Benchmark Overlap (n=314):     Mean = {np.mean(all_overlaps):.4f} (±{np.std(all_overlaps):.4f}), Median = {np.median(all_overlaps):.4f}")
print(f"Overlap Ratio (Auto / Pilot):       {np.mean(auto_overlaps) / np.mean(pilot_overlaps):.2f}x (Down from 1.82x)")

print("\n" + "="*85)
print("OVERLAP VS MRR CORRELATION (All 314 Queries)")
print("="*85)
for m in MODELS:
    r_val, p_r = stats.pearsonr(all_overlaps, mrr_per_query[m])
    rho_val, p_rho = stats.spearmanr(all_overlaps, mrr_per_query[m])
    print(f"{m:<15} | Pearson r = {r_val:.4f} (p={p_r:.4e}) | Spearman rho = {rho_val:.4f} (p={p_rho:.4e})")

print("\n" + "="*95)
print(f"{'Model':<15} | {'Scale / Query Split':<35} | {'Recall@5':<10} | {'Recall@10':<10} | {'MRR':<10} | {'nDCG@10':<10}")
print("="*95)
for m_idx, (m, label) in enumerate(zip(models, model_labels)):
    res = results_summary[m]
    p20 = res["baseline_75_pilot_queries_20_chunks"]
    p380 = res["split_75_pilot_queries_380_chunks"]
    auto = res["split_239_auto_queries_380_chunks"]
    all314 = res["overall_314_queries_380_chunks"]
    
    print(f"{label:<15} | {'Pilot Baseline (75 Qs / 20 Chunks)':<35} | {p20['recall@5']:<10.4f} | {p20['recall@10']:<10.4f} | {p20['mrr']:<10.4f} | {p20['ndcg@10']:<10.4f}")
    print(f"{'':<15} | {'Pilot Queries on Scaled (75 Qs / 380 Chunks)':<35} | {p380['recall@5']:<10.4f} | {p380['recall@10']:<10.4f} | {p380['mrr']:<10.4f} | {p380['ndcg@10']:<10.4f}")
    print(f"{'':<15} | {'Auto Queries on Scaled (239 Qs / 380 Chunks)':<35} | {auto['recall@5']:<10.4f} | {auto['recall@10']:<10.4f} | {auto['mrr']:<10.4f} | {auto['ndcg@10']:<10.4f}")
    print(f"{'':<15} | {'Overall Scaled (314 Qs / 380 Chunks)':<35} | {all314['recall@5']:<10.4f} | {all314['recall@10']:<10.4f} | {all314['mrr']:<10.4f} | {all314['ndcg@10']:<10.4f}")
    print("-" * 95)
