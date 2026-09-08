import json
import pickle
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

sys.path.append(str(Path("c:/Users/sumas/Downloads/setu-hinglish-rag-skeleton/setu-hinglish-rag")))

from setu.operators.caep import apply_caep, extract_entity_list, entity_frequencies
from setu.operators.lag import apply_lag, predict_strategy, entity_density
from setu.operators.lqp import apply_lqp
from setu.diagnosis.cmi import cmi
from setu.diagnosis.lid_entropy import lid_entropy

def main():
    chunks = []
    with open("data/processed/corpus_chunks_v2.jsonl", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
            
    doc_ids = [c["chunk_id"] for c in chunks]
    doc_text_map = {c["chunk_id"]: (c.get("title", ""), c["text"]) for c in chunks}
    doc_texts = [c["text"] for c in chunks]
    
    doc_embeddings = np.load("data/embeddings/doc_emb_bge_m3_v2.npy").astype("float32")
    faiss.normalize_L2(doc_embeddings)
    index = faiss.IndexFlatIP(doc_embeddings.shape[1])
    index.add(doc_embeddings)
    
    def faiss_search_fn(query_embedding, k=100):
        q = np.asarray(query_embedding, dtype="float32").reshape(1, -1)
        faiss.normalize_L2(q)
        scores, indices = index.search(q, k)
        return [doc_ids[i] for i in indices[0]], [float(s) for s in scores[0]]

    q_embs = np.load("data/embeddings/query_emb_bge_m3_v3.npy").astype("float32")
    
    model = SentenceTransformer("BAAI/bge-m3")
    def embed_fn(texts): return model.encode(texts, convert_to_numpy=True)
    
    entities = extract_entity_list(doc_texts)
    entity_freq = entity_frequencies(doc_texts)

    with open("results/models/caep_gate.pkl", "rb") as f: caep_gate = pickle.load(f)
    with open("results/models/lqp_model.pkl", "rb") as f: lqp_model = pickle.load(f)
    with open("results/models/lag_model_v3.pkl", "rb") as f: lag_model = pickle.load(f)

    with open("data/processed/queries_v3_final.json", "r", encoding="utf-8") as f:
        queries = json.load(f)
        
    metrics = json.load(open("results/logs/per_query_metrics_v2.json"))["bge_m3"]
    raw_failed_qids = [qid for qid, res in metrics.items() if res["hit_rate@1"] == 0.0]

    easy_queries = []
    hard_queries = []
    
    for i, q in enumerate(queries):
        qid = q.get("query_id")
        q_text = q["text"]
        target_doc = q["relevant_doc_ids"][0]

        if qid not in raw_failed_qids:
            if metrics.get(qid, {}).get("hit_rate@1") == 1.0 and len(easy_queries) < 3 and q["variant"] != "en":
                easy_queries.append({
                    "query": q_text,
                    "cmi": cmi(q_text),
                    "ops_applied": ["STOP"],
                    "doc_rank_before": 1,
                    "doc_rank_after": 1,
                    "doc_title": doc_text_map[target_doc][0],
                    "doc_text": doc_text_map[target_doc][1][:200] + "...",
                    "modified_query": q_text
                })
            continue

        # Use cached embedding for base_rank
        cached_emb = q_embs[i]
        ranked_doc_ids, _ = faiss_search_fn(cached_emb, k=20)
        try:
            base_rank = ranked_doc_ids.index(target_doc) + 1
        except ValueError:
            base_rank = 100
            
        if base_rank == 1: 
            continue # Should be raw_failed, but just in case
            
        cmi_score = cmi(q_text)
        
        # Test LQP
        projected_embedding = apply_lqp(np.asarray(cached_emb), cmi_score, lqp_model)
        lqp_ranked, _ = faiss_search_fn(projected_embedding, k=20)
        lqp_rank = lqp_ranked.index(target_doc) + 1 if target_doc in lqp_ranked else 100
        
        if lqp_rank < base_rank and lqp_rank <= 3:
            hard_queries.append({
                "query": q_text,
                "cmi": cmi_score,
                "ops_applied": ["LQP"],
                "doc_rank_before": base_rank,
                "doc_rank_after": lqp_rank,
                "doc_title": doc_text_map[target_doc][0],
                "doc_text": doc_text_map[target_doc][1][:200] + "...",
                "modified_query": "[Latent Query Projection Applied - Embedding Space Modified]"
            })
            print(f"{qid}: FIXED by LQP (Rank {base_rank} -> {lqp_rank}): {q_text}")
            if len(hard_queries) >= 3: break
            continue
            
        # Test CAEP
        caep_q = apply_caep(q_text, entities, caep_gate, entity_freq, embed_fn=embed_fn)
        if caep_q != q_text:
            caep_emb = embed_fn([caep_q])[0]
            caep_ranked, _ = faiss_search_fn(caep_emb, k=20)
            caep_rank = caep_ranked.index(target_doc) + 1 if target_doc in caep_ranked else 100
            if caep_rank < base_rank and caep_rank <= 3:
                hard_queries.append({
                    "query": q_text,
                    "cmi": cmi_score,
                    "ops_applied": ["CAEP"],
                    "doc_rank_before": base_rank,
                    "doc_rank_after": caep_rank,
                    "doc_title": doc_text_map[target_doc][0],
                    "doc_text": doc_text_map[target_doc][1][:200] + "...",
                    "modified_query": caep_q
                })
                print(f"{qid}: FIXED by CAEP (Rank {base_rank} -> {caep_rank}): {q_text}")
                if len(hard_queries) >= 3: break
                continue
                
        # Test LAG
        density = entity_density(q_text, entities)
        entropy = lid_entropy(q_text)
        strategy = predict_strategy(cmi_score, entropy, density, lag_model)
        lag_out = apply_lag(q_text, strategy, entities=entities, embed_fn=embed_fn, caep_gate=caep_gate, entity_freq=entity_freq)
        
        if isinstance(lag_out, list):
            q1_emb = embed_fn([lag_out[0]])[0]
            q2_emb = embed_fn([lag_out[1]])[0]
            r1_ids, r1_scores = faiss_search_fn(np.asarray(q1_emb))
            r2_ids, r2_scores = faiss_search_fn(np.asarray(q2_emb))
            # simplified RRF or just checking best rank
            rank1 = r1_ids.index(target_doc) + 1 if target_doc in r1_ids else 100
            rank2 = r2_ids.index(target_doc) + 1 if target_doc in r2_ids else 100
            lag_rank = min(rank1, rank2)
        else:
            if lag_out == q_text: continue
            lag_emb = embed_fn([lag_out])[0]
            lag_ranked, _ = faiss_search_fn(lag_emb, k=20)
            lag_rank = lag_ranked.index(target_doc) + 1 if target_doc in lag_ranked else 100
            
        if lag_rank < base_rank and lag_rank <= 3:
            hard_queries.append({
                "query": q_text,
                "cmi": cmi_score,
                "ops_applied": ["LAG"],
                "doc_rank_before": base_rank,
                "doc_rank_after": lag_rank,
                "doc_title": doc_text_map[target_doc][0],
                "doc_text": doc_text_map[target_doc][1][:200] + "...",
                "modified_query": f"{lag_out[0]} / {lag_out[1]}" if isinstance(lag_out, list) else lag_out
            })
            print(f"{qid}: FIXED by LAG (Rank {base_rank} -> {lag_rank}): {q_text}")
            if len(hard_queries) >= 3: break

    final_results = easy_queries[:3] + hard_queries[:3]
    with open("demo-queries.json", "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2)

if __name__ == "__main__":
    main()
