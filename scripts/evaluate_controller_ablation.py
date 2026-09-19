import json
import pickle
import sys
import time
from pathlib import Path
from collections import Counter
import random

import numpy as np
import faiss
from ranx import Qrels, Run, evaluate

sys.path.append(str(Path(__file__).resolve().parents[1]))

from setu.operators.caep import extract_entity_list, entity_frequencies
from setu.controller.setu_bandit import setu_v2_run, ACTIONS
from setu.evaluation.metrics import confidence_proxy
from setu.embeddings.loader import load_embedding_model, embed
from setu.config import set_seed

set_seed()
from setu.controller.setu_bandit import setu_v2_run, ACTIONS
from setu.evaluation.metrics import confidence_proxy
from setu.embeddings.loader import load_embedding_model, embed

class FixedStopController:
    def select_action(self, context):
        return "STOP"
    def update(self, *args, **kwargs):
        pass

class RandomController:
    def __init__(self, seed=42):
        self.rng = random.Random(seed)
        self.actions = ACTIONS
    def select_action(self, context):
        return self.rng.choice(self.actions)
    def update(self, *args, **kwargs):
        pass

def main():
    root = Path(__file__).resolve().parents[1]
    
    # 1. Load data
    chunks = []
    with open(root / "data" / "processed" / "corpus_chunks_v2.jsonl", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    doc_ids = [c["chunk_id"] for c in chunks]
    doc_texts = [c["text"] for c in chunks]
    queries = json.load(open(root / "data" / "processed" / "queries_v3_final.json", encoding="utf-8"))
    
    # 2. Embedding + FAISS
    print("Loading embedding model (BGE-M3)...")
    model = load_embedding_model("bge_m3")
    
    def embed_fn(texts):
        return embed(texts, model)
    
    doc_embeddings = np.load(root / "results" / "logs" / "doc_emb_bge_m3_v2.npy").astype("float32")
    faiss.normalize_L2(doc_embeddings)
    index = faiss.IndexFlatIP(doc_embeddings.shape[1])
    index.add(doc_embeddings)
    
    def faiss_search_fn(query_embedding, k=10):
        q = np.asarray(query_embedding, dtype="float32").reshape(1, -1)
        faiss.normalize_L2(q)
        scores, indices = index.search(q, k)
        return [doc_ids[i] for i in indices[0]], [float(s) for s in scores[0]]

    # 3. Models & entities
    entities = extract_entity_list(doc_texts)
    entity_freq = entity_frequencies(doc_texts)
    
    with open(root / "results" / "models" / "caep_gate_bge_m3.pkl", "rb") as f:
        caep_gate = pickle.load(f)
    with open(root / "results" / "models" / "lqp_model_bge_m3.pkl", "rb") as f:
        lqp_model = pickle.load(f)
    with open(root / "results" / "models" / "lag_model_v3.pkl", "rb") as f:
        lag_model = pickle.load(f)

    qrels_dict = {q["query_id"]: {d: 1 for d in q["relevant_doc_ids"]} for q in queries}
    
    fixed_stop_run = {}
    random_run = {}
    
    fixed_stop_lat = []
    random_lat = []
    fixed_stop_steps = []
    random_steps = []
    
    random_action_sequences = []
    
    controller_fs = FixedStopController()
    controller_rd = RandomController(seed=42)
    
    print(f"Evaluating {len(queries)} queries with controller ablations...")
    for i, q in enumerate(queries):
        qid, query_text = q["query_id"], q["text"]
        query_emb = embed_fn([query_text])[0]
        raw_ranking = faiss_search_fn(query_emb)
        
        # --- FIXED STOP ORACLE ---
        t0 = time.perf_counter()
        ops_fs, _, ranking_fs, _ = setu_v2_run(
            query=query_text, query_id=qid, controller=controller_fs, raw_ranking=raw_ranking, embed_fn=embed_fn,
            entities=entities, entity_freq=entity_freq, caep_gate=caep_gate,
            lqp_model=lqp_model, faiss_search_fn=faiss_search_fn, confidence_fn=confidence_proxy,
            lag_model=lag_model, train=False
        )
        fixed_stop_lat.append(time.perf_counter() - t0)
        fixed_stop_steps.append(len([o for o in ops_fs if o != "STOP"]))
        fixed_stop_run[qid] = {doc: (len(ranking_fs) - rank) for rank, doc in enumerate(ranking_fs)}
        
        # --- RANDOM POLICY ---
        t0 = time.perf_counter()
        ops_rd, _, ranking_rd, _ = setu_v2_run(
            query=query_text, query_id=qid, controller=controller_rd, raw_ranking=raw_ranking, embed_fn=embed_fn,
            entities=entities, entity_freq=entity_freq, caep_gate=caep_gate,
            lqp_model=lqp_model, faiss_search_fn=faiss_search_fn, confidence_fn=confidence_proxy,
            lag_model=lag_model, train=False
        )
        random_lat.append(time.perf_counter() - t0)
        random_steps.append(len([o for o in ops_rd if o != "STOP"]))
        random_run[qid] = {doc: (len(ranking_rd) - rank) for rank, doc in enumerate(ranking_rd)}
        random_action_sequences.append(tuple(ops_rd))
        
        if (i+1) % 50 == 0:
            print(f"  Processed {i+1}/{len(queries)} queries")
            
    # Evaluate
    qrels = Qrels(qrels_dict)
    METRICS = ["ndcg@10", "mrr", "recall@5", "recall@10"]
    
    metrics_fs = {k: float(v) for k, v in evaluate(qrels, Run(fixed_stop_run), METRICS).items()}
    metrics_fs["mean_steps"] = float(np.mean(fixed_stop_steps))
    metrics_fs["mean_latency_ms"] = float(np.mean(fixed_stop_lat) * 1000)
    
    metrics_rd = {k: float(v) for k, v in evaluate(qrels, Run(random_run), METRICS).items()}
    metrics_rd["mean_steps"] = float(np.mean(random_steps))
    metrics_rd["mean_latency_ms"] = float(np.mean(random_lat) * 1000)
    
    # Subset evaluation
    subset_qids = [q["query_id"] for q in queries if 61 <= int(q["query_id"].replace("Q", "")) <= 75]
    subset_results = {}
    if subset_qids:
        qrels_sub = Qrels({qid: qrels_dict[qid] for qid in subset_qids})
        metrics_sub_fs = {k: float(v) for k, v in evaluate(qrels_sub, Run({qid: fixed_stop_run[qid] for qid in subset_qids}), METRICS).items()}
        metrics_sub_rd = {k: float(v) for k, v in evaluate(qrels_sub, Run({qid: random_run[qid] for qid in subset_qids}), METRICS).items()}
    else:
        metrics_sub_fs = {}
        metrics_sub_rd = {}
        
    print("\nAction Sequence Diversity for Random Policy:")
    counts = Counter(random_action_sequences)
    print(f"Number of distinct patterns: {len(counts)}")
    for seq, c in counts.most_common(10):
        print(f"  {seq}: {c}")
        
    # Append to existing comparison json
    comparison_path = root / "results" / "tables" / "setu_v1_v2_comparison_scaled.json"
    if comparison_path.exists():
        with open(comparison_path, "r", encoding="utf-8") as f:
            comp_data = json.load(f)
            
        comp_data["full_dataset"]["SETU_v2_fixed_stop"] = metrics_fs
        comp_data["full_dataset"]["SETU_v2_random"] = metrics_rd
        if subset_qids:
            comp_data["misspelled_subset"]["SETU_v2_fixed_stop"] = metrics_sub_fs
            comp_data["misspelled_subset"]["SETU_v2_random"] = metrics_sub_rd
            
        comp_data["random_policy_diversity"] = {
            "n_distinct_patterns": len(counts),
            "top_patterns": {str(k): v for k, v in counts.most_common()}
        }
        
        with open(comparison_path, "w", encoding="utf-8") as f:
            json.dump(comp_data, f, indent=4)
        print(f"\nAppended controller ablation results to {comparison_path}")
    else:
        print(f"Warning: {comparison_path} not found. Cannot append.")

if __name__ == "__main__":
    main()
