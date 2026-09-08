"""
Label pilot queries with LAG sub-strategies based on empirical results.

For each query, we evaluate light_normalize, dual_variant, and full_rewrite
against the FAISS index and pick the one that yields the highest MRR.
"""
import json
import numpy as np
import sys
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer

root = Path(__file__).resolve().parents[1]
sys.path.append(str(root))

from setu.diagnosis.cmi import cmi
from setu.diagnosis.lid_entropy import lid_entropy
from setu.operators.caep import extract_entity_list
from setu.operators.lag import entity_density, SUB_STRATEGIES, light_normalize, dual_variant, full_rewrite
from setu.fusion.carf import rrf_baseline

def main():
    print("Loading corpus and queries...")
    chunks = [json.loads(line) for line in open(root / "data/processed/corpus_chunks_v2.jsonl", encoding="utf-8") if line.strip()]
    doc_ids = [c["chunk_id"] for c in chunks]
    doc_texts = [c["text"] for c in chunks]
    entities = extract_entity_list(doc_texts)
    
    queries = json.load(open(root / "data/processed/queries_v3_final.json", encoding="utf-8"))
    
    print("Loading BGE-M3 and precomputed FAISS index...")
    model = SentenceTransformer("BAAI/bge-m3")
    
    def embed_fn(texts):
        return model.encode(texts, convert_to_numpy=True)
        
    doc_embeddings = np.load(root / "data/embeddings/doc_emb_bge_m3_v2.npy").astype("float32")
    faiss.normalize_L2(doc_embeddings)
    index = faiss.IndexFlatIP(doc_embeddings.shape[1])
    index.add(doc_embeddings)
    
    def faiss_search_fn(query_embedding, k=10):
        q = np.asarray(query_embedding, dtype="float32").reshape(1, -1)
        faiss.normalize_L2(q)
        scores, indices = index.search(q, k)
        ranked_doc_ids = [doc_ids[i] for i in indices[0]]
        return ranked_doc_ids

    labeled_queries = []
    strategy_counts = {s: 0 for s in SUB_STRATEGIES}
    
    print("Evaluating sub-strategies to find empirical best labels...")
    for idx, q in enumerate(queries):
        qid = q["query_id"]
        text = q["text"]
        relevant = q.get("relevant_doc_ids", [])
        
        cmi_score = cmi(text)
        entropy_score = lid_entropy(text)
        density = entity_density(text, entities)
        
        strats_mrr = {}
        
        for strat in SUB_STRATEGIES:
            if strat == "light_normalize":
                out = light_normalize(text)
                ranked = faiss_search_fn(embed_fn([out])[0])
                mrr = sum(1.0 / (i + 1) for i, r in enumerate(ranked) if r in relevant)
                strats_mrr[strat] = mrr
                
            elif strat == "dual_variant":
                out = dual_variant(text, entities)
                r1 = faiss_search_fn(embed_fn([out[0]])[0])
                r2 = faiss_search_fn(embed_fn([out[1]])[0])
                ranked = rrf_baseline([r1, r2])
                mrr = sum(1.0 / (i + 1) for i, r in enumerate(ranked) if r in relevant)
                strats_mrr[strat] = mrr
                
            elif strat == "full_rewrite":
                out = full_rewrite(text, entities=entities)
                ranked = faiss_search_fn(embed_fn([out])[0])
                mrr = sum(1.0 / (i + 1) for i, r in enumerate(ranked) if r in relevant)
                strats_mrr[strat] = mrr
                
        # Best strategy wins. If tie, fallback to light_normalize (0).
        best_strat = "light_normalize"
        best_mrr = -1
        for s in SUB_STRATEGIES:
            if strats_mrr[s] > best_mrr:
                best_mrr = strats_mrr[s]
                best_strat = s
                
        strategy_counts[best_strat] += 1
        label_idx = SUB_STRATEGIES.index(best_strat)
        
        labeled_queries.append({
            "query_id": qid,
            "text": text,
            "relevant_doc_ids": relevant,
            "cmi": float(cmi_score),
            "lid_entropy": float(entropy_score),
            "entity_density": float(density),
            "strategy": best_strat,
            "label": label_idx,
        })
        
        if (idx+1) % 50 == 0:
            print(f"Processed {idx+1}/{len(queries)} queries...")
        
    out_path = root / "data/processed/lag_labels_v3.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(labeled_queries, f, indent=2, ensure_ascii=False)
        
    print(f"Saved {len(labeled_queries)} labeled queries -> {out_path}")
    print(f"Empirical Strategy Distribution: {strategy_counts}")

if __name__ == "__main__":
    main()
