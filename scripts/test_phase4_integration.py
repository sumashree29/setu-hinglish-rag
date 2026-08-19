"""
Phase 4 Task 5 -- integration test: run setu_v1_fixed_order() and
setu_v2_run() end-to-end against REAL pilot corpus + real fitted LQP model.

CAEP gate is a PLACEHOLDER, bootstrapped on synthetic labeled examples --
NOT R2's real trained gate, since no labeled CAEP training data/script exists
in the repo yet. Swap in the real fitted gate once R2 provides one.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

sys.path.append(str(Path(__file__).resolve().parents[1]))

from setu.operators.caep import extract_entity_list, entity_frequencies, fit_caep_gate
from setu.operators.lqp import fit_lqp
from setu.controller.setu_bandit import setu_v1_fixed_order, setu_v2_run, EpsilonGreedyController
from setu.evaluation.metrics import confidence_proxy

# --- Load real pilot corpus ---
chunks = []
with open("data/processed/corpus_chunks.jsonl", encoding="utf-8") as f:
    for line in f:
        chunks.append(json.loads(line))
doc_ids = [c["chunk_id"] for c in chunks]
doc_texts = [c["text"] for c in chunks]

queries = json.load(open("data/processed/queries_remapped.json", encoding="utf-8"))

# --- Embedding function (Indic-SBERT, already cached locally from Phase 1) ---
print("Loading embedding model...")
model = SentenceTransformer("l3cube-pune/indic-sentence-similarity-sbert")


def embed_fn(texts):
    return model.encode(texts, convert_to_numpy=True)


# --- FAISS index over real corpus ---
doc_embeddings = embed_fn(doc_texts).astype("float32")
faiss.normalize_L2(doc_embeddings)
index = faiss.IndexFlatIP(doc_embeddings.shape[1])
index.add(doc_embeddings)


def faiss_search_fn(query_embedding, k=4):
    q = np.asarray(query_embedding, dtype="float32").reshape(1, -1)
    faiss.normalize_L2(q)
    scores, indices = index.search(q, k)
    ranked_doc_ids = [doc_ids[i] for i in indices[0]]
    ranked_scores = [float(s) for s in scores[0]]
    return ranked_doc_ids, ranked_scores


# --- Real entities from real corpus ---
entities = extract_entity_list(doc_texts)
entity_freq = entity_frequencies(doc_texts)
print(f"Extracted {len(entities)} entities from real corpus: {entities[:10]}")

# --- PLACEHOLDER CAEP gate (bootstrapped, NOT R2's real trained gate) ---
placeholder_features = [
    [100.0, 1.0, 5.0], [95.0, 0.95, 3.0], [90.0, 0.9, 2.0],  # clear matches -> preserve
    [40.0, 0.3, 0.0], [30.0, 0.2, 0.0], [20.0, 0.1, 0.0],    # weak matches -> substitute
]
placeholder_labels = [1, 1, 1, 0, 0, 0]
caep_gate = fit_caep_gate(placeholder_features, placeholder_labels)
print("WARNING: CAEP gate is a placeholder bootstrapped on synthetic data, not R2's real trained gate\n")

# --- Real LQP model, fit on real PHINC data ---
print("Loading PHINC and fitting LQP...")
phinc = pd.read_csv("data/raw/hinge_phinc/phinc.csv")
sample = phinc.sample(n=min(200, len(phinc)), random_state=42)  # small sample for speed
X = embed_fn(sample["Sentence"].tolist())
Y = embed_fn(sample["English_Translation"].tolist())
lqp_model = fit_lqp(X, Y, alpha_reg=1.0)
print(f"LQP model fit on {len(sample)} real PHINC pairs\n")

# --- Run v1 baseline on a few real queries ---
print("=== SETU v1 (fixed order) on 3 real queries ===")
high_cmi_queries = sorted(queries, key=lambda q: -abs(0.5 - 0))[:0]  # placeholder, replaced below
from setu.diagnosis.cmi import cmi as _cmi_fn
queries_with_cmi = [(q, _cmi_fn(q["text"])) for q in queries]
high_cmi_queries = [q for q, c in sorted(queries_with_cmi, key=lambda x: -x[1])[:3]]
for q in high_cmi_queries:
    query_text = q["text"]
    query_emb = embed_fn([query_text])[0]
    raw_ranking = faiss_search_fn(query_emb)

    result = setu_v1_fixed_order(
        query=query_text,
        raw_ranking=raw_ranking,
        embed_fn=embed_fn,
        entities=entities,
        entity_freq=entity_freq,
        caep_gate=caep_gate,
        lqp_model=lqp_model,
        faiss_search_fn=faiss_search_fn,
    )
    print(f"\nQuery: {query_text}")
    print(f"  expected: {q['relevant_doc_ids']}")
    print(f"  RAW (uncorrected) top2:  {raw_ranking[0][:2]}")
    print(f"  CORRECTED (v1) top2:     {result['final_ranking'][:2]}")
# --- Run v2 (epsilon-greedy) on the same queries ---
print("\n=== SETU v2 (epsilon-greedy) on 3 real queries ===")
controller = EpsilonGreedyController(epsilon=0.2)
high_cmi_queries = sorted(queries, key=lambda q: -abs(0.5 - 0))[:0]  # placeholder, replaced below
from setu.diagnosis.cmi import cmi as _cmi_fn
queries_with_cmi = [(q, _cmi_fn(q["text"])) for q in queries]
high_cmi_queries = [q for q, c in sorted(queries_with_cmi, key=lambda x: -x[1])[:3]]
for q in high_cmi_queries:
    query_text = q["text"]
    query_emb = embed_fn([query_text])[0]
    raw_ranking = faiss_search_fn(query_emb)

    ops, conf_trace = setu_v2_run(
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
    )
    print(f"\nQuery: {query_text}")
    print(f"  operators used: {ops}")
    print(f"  confidence trace: {[round(c,3) for c in conf_trace]}")

print("\nIntegration test complete.")