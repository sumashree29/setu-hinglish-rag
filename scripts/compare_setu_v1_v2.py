"""
Phase 4 Task 6: SETU v1 (fixed order) vs v2 (LinUCB) vs raw baseline,
across all 75 pilot queries (60 original + 15 misspelled-entity subset).
Reports Recall@k/MRR/nDCG, mean steps, latency.

Uses real fitted CAEP gate, LQP model, and LAG model (results/models/*.pkl).
v2 uses LinUCBController (real contextual bandit); switch to
EpsilonGreedyController below if you want the simpler baseline policy instead.
"""
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from ranx import Qrels, Run, evaluate

sys.path.append(str(Path(__file__).resolve().parents[1]))

from setu.operators.caep import extract_entity_list, entity_frequencies
from setu.controller.setu_bandit import setu_v1_fixed_order, setu_v2_run, LinUCBController, EpsilonGreedyController
from setu.evaluation.metrics import confidence_proxy

# --- Load real pilot corpus + queries ---
chunks = []
with open("data/processed/corpus_chunks.jsonl", encoding="utf-8") as f:
    for line in f:
        chunks.append(json.loads(line))
doc_ids = [c["chunk_id"] for c in chunks]
doc_texts = [c["text"] for c in chunks]
queries = json.load(open("data/processed/queries_remapped.json", encoding="utf-8"))

# --- Embedding + FAISS setup (BGE-M3, matches fitted LQP model dim 1024) ---
print("Loading embedding model (BGE-M3)...")
model = SentenceTransformer("BAAI/bge-m3")


def embed_fn(texts):
    return model.encode(texts, convert_to_numpy=True)


doc_embeddings = embed_fn(doc_texts).astype("float32")
faiss.normalize_L2(doc_embeddings)
index = faiss.IndexFlatIP(doc_embeddings.shape[1])
index.add(doc_embeddings)


def faiss_search_fn(query_embedding, k=10):
    q = np.asarray(query_embedding, dtype="float32").reshape(1, -1)
    faiss.normalize_L2(q)
    scores, indices = index.search(q, k)
    return [doc_ids[i] for i in indices[0]], [float(s) for s in scores[0]]


# --- Entities + fitted models ---
entities = extract_entity_list(doc_texts)
entity_freq = entity_frequencies(doc_texts)

print("Loading real fitted CAEP gate, LQP model, and LAG model from results/models/...")
with open("results/models/caep_gate.pkl", "rb") as f:
    caep_gate = pickle.load(f)
with open("results/models/lqp_model.pkl", "rb") as f:
    lqp_model = pickle.load(f)
with open("results/models/lag_model.pkl", "rb") as f:
    lag_model = pickle.load(f)
print("Loaded caep_gate.pkl, lqp_model.pkl, and lag_model.pkl (trained on real corpus/PHINC/pilot labels)\n")

# --- Run all 3 systems across all queries ---
qrels_dict = {q["query_id"]: {d: 1 for d in q["relevant_doc_ids"]} for q in queries}

raw_run, v1_run, v2_run = {}, {}, {}
v1_latencies, v2_latencies = [], []
v2_step_counts = []

controller = LinUCBController(context_dim=7, alpha=1.0)

print(f"Running {len(queries)} queries through raw / v1 / v2...")
for i, q in enumerate(queries):
    qid, query_text = q["query_id"], q["text"]
    query_emb = embed_fn([query_text])[0]
    raw_ranking = faiss_search_fn(query_emb)

    raw_run[qid] = {doc: score for doc, score in zip(raw_ranking[0], raw_ranking[1])}

    t0 = time.perf_counter()
    v1_result = setu_v1_fixed_order(
        query=query_text, raw_ranking=raw_ranking, embed_fn=embed_fn,
        entities=entities, entity_freq=entity_freq, caep_gate=caep_gate,
        lqp_model=lqp_model, faiss_search_fn=faiss_search_fn, lag_model=lag_model,
    )
    v1_latencies.append(time.perf_counter() - t0)
    v1_ranking = v1_result["final_ranking"]
    v1_run[qid] = {doc: (len(v1_ranking) - rank) for rank, doc in enumerate(v1_ranking)}

    t0 = time.perf_counter()
    ops, conf_trace, v2_ranking = setu_v2_run(
        query=query_text, controller=controller, raw_ranking=raw_ranking, embed_fn=embed_fn,
        entities=entities, entity_freq=entity_freq, caep_gate=caep_gate,
        lqp_model=lqp_model, faiss_search_fn=faiss_search_fn, confidence_fn=confidence_proxy,
        lag_model=lag_model,
    )

    v2_latencies.append(time.perf_counter() - t0)
    v2_step_counts.append(len([o for o in ops if o != "STOP"]))
    v2_run[qid] = {doc: (len(v2_ranking) - rank) for rank, doc in enumerate(v2_ranking)}

    if (i + 1) % 15 == 0:
        print(f"  {i+1}/{len(queries)} done...")

print("\nAll queries processed.\n")

# --- Metrics across full dataset ---
qrels = Qrels(qrels_dict)
METRICS = ["ndcg@10", "mrr", "recall@5", "recall@10"]

raw_metrics = {k: float(v) for k, v in evaluate(qrels, Run(raw_run), METRICS).items()}
v1_metrics = {k: float(v) for k, v in evaluate(qrels, Run(v1_run), METRICS).items()}
v1_metrics["mean_latency_ms"] = float(np.mean(v1_latencies) * 1000)
v2_metrics = {k: float(v) for k, v in evaluate(qrels, Run(v2_run), METRICS).items()}
v2_metrics["mean_steps"] = float(np.mean(v2_step_counts))
v2_metrics["mean_latency_ms"] = float(np.mean(v2_latencies) * 1000)

print(f"==================================================")
print(f"=== FULL DATASET ({len(queries)} Queries) ===")
print(f"==================================================")
print("=== RAW baseline ===")
print(raw_metrics)

print("\n=== SETU v1 (fixed order) ===")
print(v1_metrics)
print(f"Mean latency: {v1_metrics['mean_latency_ms']:.2f} ms")

print("\n=== SETU v2 (LinUCB) ===")
print(v2_metrics)
print(f"Mean steps taken: {v2_metrics['mean_steps']:.2f}")
print(f"Mean latency: {v2_metrics['mean_latency_ms']:.2f} ms")

# --- Metrics across Q61-Q75 subset (Misspelled entity queries only) ---
subset_qids = [q["query_id"] for q in queries if int(q["query_id"].replace("Q", "")) >= 61]
subset_results = {}
if subset_qids:
    qrels_sub = Qrels({qid: qrels_dict[qid] for qid in subset_qids})
    raw_run_sub = Run({qid: raw_run[qid] for qid in subset_qids})
    v1_run_sub = Run({qid: v1_run[qid] for qid in subset_qids})
    v2_run_sub = Run({qid: v2_run[qid] for qid in subset_qids})

    raw_sub_metrics = {k: float(v) for k, v in evaluate(qrels_sub, raw_run_sub, METRICS).items()}
    v1_sub_metrics = {k: float(v) for k, v in evaluate(qrels_sub, v1_run_sub, METRICS).items()}
    v2_sub_metrics = {k: float(v) for k, v in evaluate(qrels_sub, v2_run_sub, METRICS).items()}

    subset_results = {
        "RAW": raw_sub_metrics,
        "SETU_v1": v1_sub_metrics,
        "SETU_v2": v2_sub_metrics,
    }

    print(f"\n==================================================")
    print(f"=== MISSPELLED ENTITY SUBSET ({len(subset_qids)} Queries: Q61-Q75) ===")
    print(f"==================================================")
    print("=== RAW baseline (misspelled subset) ===")
    print(raw_sub_metrics)

    print("\n=== SETU v1 (misspelled subset) ===")
    print(v1_sub_metrics)

    print("\n=== SETU v2 (misspelled subset) ===")
    print(v2_sub_metrics)

# --- Save results to disk ---
tables_dir = Path(__file__).resolve().parents[1] / "results" / "tables"
tables_dir.mkdir(parents=True, exist_ok=True)
comparison_out = {
    "full_dataset": {
        "RAW": raw_metrics,
        "SETU_v1": v1_metrics,
        "SETU_v2": v2_metrics,
    },
    "misspelled_subset": subset_results,
    "n_total_queries": len(queries),
    "n_misspelled_queries": len(subset_qids),
}
comparison_path = tables_dir / "setu_v1_v2_comparison.json"
with open(comparison_path, "w", encoding="utf-8") as f:
    json.dump(comparison_out, f, indent=2)
print(f"\nSaved comparison table to {comparison_path}")

