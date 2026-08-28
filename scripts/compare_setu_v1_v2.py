"""
Phase 4 Task 6: SETU v1 (fixed order) vs v2 (epsilon-greedy) vs raw baseline,
across all 60 pilot queries. Reports Recall@k/MRR/nDCG, mean steps, latency.

CAVEAT: CAEP gate is still the synthetic placeholder from Task 5 (no real
labeled training data from R2 yet). Treat these as pipeline-validation
numbers, not final paper results -- rerun once the real gate exists.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from ranx import Qrels, Run, evaluate

sys.path.append(str(Path(__file__).resolve().parents[1]))

from setu.operators.caep import extract_entity_list, entity_frequencies, fit_caep_gate
from setu.operators.lqp import fit_lqp
from setu.controller.setu_bandit import setu_v1_fixed_order, setu_v2_run, EpsilonGreedyController
from setu.evaluation.metrics import confidence_proxy

# --- Load real pilot corpus + queries ---
chunks = []
with open("data/processed/corpus_chunks.jsonl", encoding="utf-8") as f:
    for line in f:
        chunks.append(json.loads(line))
doc_ids = [c["chunk_id"] for c in chunks]
doc_texts = [c["text"] for c in chunks]
queries = json.load(open("data/processed/queries_remapped.json", encoding="utf-8"))

# --- Embedding + FAISS setup ---
print("Loading embedding model...")
model = SentenceTransformer("l3cube-pune/indic-sentence-similarity-sbert")


def embed_fn(texts):
    return model.encode(texts, convert_to_numpy=True)


doc_embeddings = embed_fn(doc_texts).astype("float32")
faiss.normalize_L2(doc_embeddings)
index = faiss.IndexFlatIP(doc_embeddings.shape[1])
index.add(doc_embeddings)


def faiss_search_fn(query_embedding, k=4):
    q = np.asarray(query_embedding, dtype="float32").reshape(1, -1)
    faiss.normalize_L2(q)
    scores, indices = index.search(q, k)
    return [doc_ids[i] for i in indices[0]], [float(s) for s in scores[0]]


# --- Entities + placeholder CAEP gate ---
entities = extract_entity_list(doc_texts)
entity_freq = entity_frequencies(doc_texts)
placeholder_features = [[100.0, 1.0, 5.0], [95.0, 0.95, 3.0], [90.0, 0.9, 2.0],
                         [40.0, 0.3, 0.0], [30.0, 0.2, 0.0], [20.0, 0.1, 0.0]]
placeholder_labels = [1, 1, 1, 0, 0, 0]
caep_gate = fit_caep_gate(placeholder_features, placeholder_labels)
print("WARNING: using placeholder CAEP gate, not R2's real trained gate\n")

# --- Real LQP model, fit on real PHINC data ---
print("Fitting LQP on PHINC...")
phinc = pd.read_csv("data/raw/hinge_phinc/phinc.csv")
sample = phinc.sample(n=min(3000, len(phinc)), random_state=42)
X = embed_fn(sample["Sentence"].tolist())
Y = embed_fn(sample["English_Translation"].tolist())
lqp_model = fit_lqp(X, Y, alpha_reg=1.0)
print(f"LQP fit on {len(sample)} PHINC pairs\n")

# --- Run all 3 systems across all 60 queries ---
qrels_dict = {q["query_id"]: {d: 1 for d in q["relevant_doc_ids"]} for q in queries}

raw_run, v1_run, v2_run = {}, {}, {}
v1_latencies, v2_latencies = [], []
v2_step_counts = []

controller = EpsilonGreedyController(epsilon=0.15)

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
        lqp_model=lqp_model, faiss_search_fn=faiss_search_fn,
    )
    v1_latencies.append(time.perf_counter() - t0)
    v1_ranking = v1_result["final_ranking"]
    v1_run[qid] = {doc: (len(v1_ranking) - rank) for rank, doc in enumerate(v1_ranking)}

    t0 = time.perf_counter()
    ops, conf_trace, v2_ranking = setu_v2_run(
        query=query_text, controller=controller, raw_ranking=raw_ranking, embed_fn=embed_fn,
        entities=entities, entity_freq=entity_freq, caep_gate=caep_gate,
        lqp_model=lqp_model, faiss_search_fn=faiss_search_fn, confidence_fn=confidence_proxy,
    )
    
    v2_latencies.append(time.perf_counter() - t0)
    v2_step_counts.append(len([o for o in ops if o != "STOP"]))
    v2_run[qid] = {doc: (len(v2_ranking) - rank) for rank, doc in enumerate(v2_ranking)}

    if (i + 1) % 15 == 0:
        print(f"  {i+1}/{len(queries)} done...")

print("\nAll queries processed.\n")

# --- Metrics ---
qrels = Qrels(qrels_dict)
METRICS = ["ndcg@10", "mrr", "recall@5", "recall@10"]

print("=== RAW baseline ===")
print(evaluate(qrels, Run(raw_run), METRICS))

print("\n=== SETU v1 (fixed order) ===")
print(evaluate(qrels, Run(v1_run), METRICS))
print(f"Mean latency: {np.mean(v1_latencies)*1000:.2f} ms")

print("\n=== SETU v2 (epsilon-greedy) ===")
print(evaluate(qrels, Run(v2_run), METRICS))
print(f"Mean steps taken: {np.mean(v2_step_counts):.2f}")
print(f"Mean latency: {np.mean(v2_latencies)*1000:.2f} ms")