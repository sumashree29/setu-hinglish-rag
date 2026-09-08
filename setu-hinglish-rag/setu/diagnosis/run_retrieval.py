"""
For each embedding model: build a FAISS index over the 20 docs, retrieve
top-k for all 60 queries, and save the ranked results to JSON.
"""
import json
import sys
from pathlib import Path

import numpy as np
import faiss

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config

LOG_DIR = config.ROOT / "results" / "logs"

query_ids = json.load(open(LOG_DIR / "query_ids.json"))
doc_ids = json.load(open(LOG_DIR / "doc_ids.json"))

MODEL_KEYS = ["indic_sbert", "bge_m3", "me5_large"]

all_results = {}

for model_key in MODEL_KEYS:
    print(f"\n=== Retrieval with {model_key} ===")
    q_emb = np.load(LOG_DIR / f"query_emb_{model_key}.npy").astype("float32")
    d_emb = np.load(LOG_DIR / f"doc_emb_{model_key}.npy").astype("float32")

    # Normalize for cosine similarity via inner product
    faiss.normalize_L2(q_emb)
    faiss.normalize_L2(d_emb)

    index = faiss.IndexFlatIP(d_emb.shape[1])
    index.add(d_emb)

    top_k = config.TOP_K
    scores, indices = index.search(q_emb, top_k)

    model_results = {}
    for i, qid in enumerate(query_ids):
        ranked_doc_ids = [doc_ids[idx] for idx in indices[i]]
        ranked_scores = [float(s) for s in scores[i]]
        model_results[qid] = {
            "ranked_doc_ids": ranked_doc_ids,
            "scores": ranked_scores,
        }

    all_results[model_key] = model_results
    print(f"Retrieved top-{top_k} for {len(query_ids)} queries.")

with open(LOG_DIR / "retrieval_results.json", "w") as f:
    json.dump(all_results, f, indent=2)

print("\nSaved retrieval results to", LOG_DIR / "retrieval_results.json")

# Quick spot check: show top-3 for the first query, for each model
print("\n--- Spot check: first query, top-3 per model ---")
for model_key in MODEL_KEYS:
    r = all_results[model_key][query_ids[0]]
    print(f"{model_key}: {r['ranked_doc_ids'][:3]}  scores={[round(s,3) for s in r['scores'][:3]]}")