import sys
import json
import pickle
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
try:
    from setu.config import set_seed
    set_seed()
except ImportError:
    pass

import numpy as np
import faiss

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from setu.operators.caep import extract_entity_list, entity_frequencies
from setu.controller.setu_bandit import setu_v2_run, LinUCBController
from setu.evaluation.metrics import confidence_proxy
from setu.embeddings.loader import load_embedding_model, embed

def main():
    print("Loading queries and corpus...")
    queries = json.load(open(ROOT / "data" / "processed" / "queries_v3_final.json", encoding="utf-8"))
    chunks = [json.loads(line) for line in open(ROOT / "data" / "processed" / "corpus_chunks_v2.jsonl", encoding="utf-8") if line.strip()]
    
    doc_ids = [c["chunk_id"] for c in chunks]
    doc_texts = [c["text"] for c in chunks]

    print("Loading BGE-M3 and encoding corpus...")
    model = load_embedding_model("bge_m3")
    def embed_fn(texts):
        return embed(texts, model).astype("float32")
        
    doc_emb = embed_fn(doc_texts)
    faiss.normalize_L2(doc_emb)
    idx = faiss.IndexFlatIP(doc_emb.shape[1])
    idx.add(doc_emb)

    def faiss_search_fn(query_embedding, k=10):
        q = np.asarray(query_embedding, dtype="float32").reshape(1, -1)
        faiss.normalize_L2(q)
        scores, indices = idx.search(q, k)
        return [doc_ids[i] for i in indices[0]], [float(score) for score in scores[0]]

    entities = extract_entity_list(doc_texts)
    entity_freq = entity_frequencies(doc_texts)

    print("Loading operators...")
    caep_gate = pickle.load(open(ROOT / "results" / "models" / "caep_gate_bge_m3.pkl", "rb"))
    lqp_model = pickle.load(open(ROOT / "results" / "models" / "lqp_model_bge_m3.pkl", "rb"))
    lag_model = pickle.load(open(ROOT / "results" / "models" / "lag_model_v3.pkl", "rb"))

    print("Initializing LinUCBController for exploration (alpha=1.0)...")
    set_seed()
    controller = LinUCBController(context_dim=7, alpha=1.0)
    
    trajectory_path = ROOT / "data" / "logs" / "trajectories_v3.jsonl"
    if trajectory_path.exists():
        import time
        import shutil
        archive_dir = ROOT / "results" / "archive"
        archive_dir.mkdir(parents=True, exist_ok=True)
        timestamp = int(time.time())
        archive_path = archive_dir / f"trajectories_v3_{timestamp}.jsonl"
        shutil.move(trajectory_path, archive_path)
        print(f"Archived existing trajectory log to {archive_path}")

    print(f"Generating trajectories for {len(queries)} queries...")
    for i, q in enumerate(queries):
        q_txt = q["text"]
        q_id = q["query_id"]
        q_emb = embed_fn([q_txt])[0]
        raw_ranking = faiss_search_fn(q_emb)

        # setu_v2_run(train=True) logs trajectory to data/logs/trajectories_v3.jsonl automatically
        setu_v2_run(
            query=q_txt,
            query_id=q_id,
            controller=controller,
            raw_ranking=raw_ranking,
            embed_fn=embed_fn,
            entities=entities,
            entity_freq=entity_freq,
            caep_gate=caep_gate,
            lqp_model=lqp_model,
            faiss_search_fn=faiss_search_fn,
            confidence_fn=confidence_proxy,
            max_steps=4,
            lag_model=lag_model,
            train=True
        )
        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(queries)} queries.")
            
    print("Trajectory generation complete.")

if __name__ == "__main__":
    main()
