import json
import pickle
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore') # ignore scikit-learn warnings

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Setup path
sys.path.append(str(Path(__file__).resolve().parent))

from setu.operators.caep import extract_entity_list, entity_frequencies
from setu.controller.setu_bandit import setu_v2_run, LinUCBController
from setu.evaluation.metrics import confidence_proxy

print("Initializing SETU Interactive Tester...")
print("[1/4] Loading corpus...")
chunks = []
with open("data/processed/corpus_chunks_v2.jsonl", encoding="utf-8") as f:
    for line in f:
        chunks.append(json.loads(line))

# Using a subset of chunks if too large to make it fast? No, let's load all of them.
doc_ids = [c["chunk_id"] for c in chunks]
doc_texts = [c["text"] for c in chunks]

# We need a small doc_id to text mapping for printing the actual text
doc_text_map = {c["chunk_id"]: c["text"] for c in chunks}

print("[2/4] Loading embedding model (BGE-M3) and building index...")
model = SentenceTransformer("BAAI/bge-m3")

def embed_fn(texts):
    return model.encode(texts, convert_to_numpy=True)

# Build FAISS index using precomputed embeddings for instant load!
doc_embeddings = np.load("data/embeddings/doc_emb_bge_m3_v2.npy").astype("float32")
faiss.normalize_L2(doc_embeddings)
index = faiss.IndexFlatIP(doc_embeddings.shape[1])
index.add(doc_embeddings)

def faiss_search_fn(query_embedding, k=3):
    q = np.asarray(query_embedding, dtype="float32").reshape(1, -1)
    faiss.normalize_L2(q)
    scores, indices = index.search(q, k)
    ranked_doc_ids = [doc_ids[i] for i in indices[0]]
    ranked_scores = [float(s) for s in scores[0]]
    return ranked_doc_ids, ranked_scores

print("[3/4] Extracting entities and loading operators...")
entities = extract_entity_list(doc_texts)
entity_freq = entity_frequencies(doc_texts)

with open("results/models/caep_gate.pkl", "rb") as f:
    caep_gate = pickle.load(f)
with open("results/models/lqp_model.pkl", "rb") as f:
    lqp_model = pickle.load(f)
with open("results/models/lag_model_v3.pkl", "rb") as f:
    lag_model = pickle.load(f)

print("[4/4] Loading SETU v2 Bandit Controller...")
try:
    controller = LinUCBController.load("results/models/linucb_policy.pkl")
except FileNotFoundError:
    print("Warning: LinUCB policy not found. Falling back to a random/epsilon-greedy controller for demo.")
    from setu.controller.setu_bandit import EpsilonGreedyController
    controller = EpsilonGreedyController(epsilon=0.0)

print("\n" + "="*50)
print("SYSTEM READY! Type 'exit' to quit.")
print("="*50 + "\n")

while True:
    query_text = input("Enter your Hinglish query: ")
    if query_text.lower() in ['exit', 'quit']:
        break
    
    if not query_text.strip():
        continue
        
    print("\n--- Running SETU Pipeline ---")
    
    # 1. Base embedding and raw ranking
    query_emb = embed_fn([query_text])[0]
    raw_ranking = faiss_search_fn(query_emb, k=3)
    
    # 2. Run SETU Controller
    ops, conf_trace, v2_ranking = setu_v2_run(
        query=query_text,
        controller=controller,
        raw_ranking=raw_ranking,
        embed_fn=embed_fn,
        entities=entities,
        entity_freq=entity_freq,
        caep_gate=caep_gate,
        lqp_model=lqp_model,
        faiss_search_fn=faiss_search_fn,
        confidence_fn=confidence_proxy,
        lag_model=lag_model,
        train=False # Don't update weights during testing
    )
    
    print(f"\n[SETU Controller Actions]: {ops}")
    print(f"[Confidence Trace]: {[round(c, 3) for c in conf_trace]}")
    
    print("\n--- Top 3 Retrieved Documents (SETU v2) ---")
    for i, doc_id in enumerate(v2_ranking[:3]):
        print(f"\n[{i+1}] (Doc ID: {doc_id})")
        print(f"Content: {doc_text_map.get(doc_id, 'Unknown')}")
    
    print("\n" + "-"*50 + "\n")
