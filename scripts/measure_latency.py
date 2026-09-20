import json
import pickle
import sys
import time
from pathlib import Path
import numpy as np
import faiss
import random
import platform
import psutil
import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from setu.operators.caep import extract_entity_list, entity_frequencies
from setu.controller.setu_bandit import setu_v1_fixed_order, setu_v2_run, LinUCBController
from setu.evaluation.metrics import confidence_proxy
from setu.embeddings.loader import load_embedding_model, embed
from setu.config import set_seed
from setu.operators.lag import fit_lag_v2

set_seed()

def get_hardware_info():
    return {
        "os": platform.platform(),
        "cpu_processor": platform.processor(),
        "cpu_cores_physical": psutil.cpu_count(logical=False),
        "cpu_cores_logical": psutil.cpu_count(logical=True),
        "ram_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "gpu": "None used (Local CPU run)",
        "python_version": platform.python_version(),
        "packages": {
            "numpy": np.__version__,
            "torch": torch.__version__,
            "faiss": faiss.__version__
        }
    }

def compute_stats(latencies):
    arr = np.array(latencies) * 1000  # ms
    return {
        "mean_ms": float(np.mean(arr)),
        "median_ms": float(np.median(arr)),
        "sd_ms": float(np.std(arr)),
        "p95_ms": float(np.percentile(arr, 95))
    }

def split_trajectories_by_fold(trajectories, train_ids, test_ids):
    train_ids = set(train_ids)
    test_ids = set(test_ids)
    train_traj = [r for r in trajectories if r.get("query_id") in train_ids]
    test_traj = [r for r in trajectories if r.get("query_id") in test_ids]
    return train_traj, test_traj

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
        
    with open(root / "data" / "processed" / "lag_labels_v3.json", "r", encoding="utf-8") as f:
        lag_labels_data = json.load(f)

    # 4. Trajectories for fold controllers (v2)
    trajectories = []
    with open(root / "data" / "logs" / "trajectories_v3.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                trajectories.append(json.loads(line))
    
    qids = [q["query_id"] for q in queries]
    q_by_id = {q["query_id"]: q for q in queries}
    folds = np.array_split(qids, 5)
    
    fold_controllers = {}
    fold_lag_models = {}
    for test_qids in folds:
        train_ids = set(qid for qid in qids if qid not in test_qids)
        test_ids = set(test_qids)
        
        train_traj, _ = split_trajectories_by_fold(trajectories, train_ids, test_ids)
        
        fold_lag_data = [d for d in lag_labels_data if d["query_id"] in train_ids]
        X_lag = np.array([[d["cmi"], d["lid_entropy"], d["entity_density"]] for d in fold_lag_data])
        y_lag = np.array([d["label"] for d in fold_lag_data])
        fold_lag_model = fit_lag_v2(X_lag, y_lag)
        
        fold_controller = LinUCBController(context_dim=7, alpha=0.0)
        current_query_id = None
        tried = {"LAG": 0.0, "CAEP": 0.0, "LQP": 0.0}
        for row in train_traj:
            q_id = row.get("query_id")
            step_val = float(row.get("state", {}).get("step", 0))
            if q_id != current_query_id or step_val == 0:
                current_query_id = q_id
                tried = {"LAG": 0.0, "CAEP": 0.0, "LQP": 0.0}

            cmi_val = float(row["state"]["cmi"])
            entropy_val = float(row["state"]["lid_entropy"])
            conf_val = float(row["state"]["confidence"])
            context = np.array([
                cmi_val, entropy_val, conf_val, step_val,
                tried["LAG"], tried["CAEP"], tried["LQP"]
            ], dtype=float)
            action = row["action"]
            reward = float(row.get("reward", 0.0))
            fold_controller.update(context, action, reward)
            if action in tried:
                tried[action] = 1.0
                
        for qid in test_qids:
            fold_controllers[qid] = fold_controller
            fold_lag_models[qid] = fold_lag_model

    # 5. Warm-up
    print("Performing 10 warm-up queries (discarded)...")
    warmup_queries = queries[:10]
    for q in warmup_queries:
        query_text = q["text"]
        qid = q["query_id"]
        # Raw
        q_emb = embed_fn([query_text])[0]
        raw_ranking = faiss_search_fn(q_emb)
        # v1
        setu_v1_fixed_order(
            query=query_text, raw_ranking=raw_ranking, embed_fn=embed_fn,
            entities=entities, entity_freq=entity_freq, caep_gate=caep_gate,
            lqp_model=lqp_model, faiss_search_fn=faiss_search_fn, lag_model=fold_lag_models[qid],
        )
        # v2
        setu_v2_run(
            query=query_text, query_id=qid, controller=fold_controllers[qid], raw_ranking=raw_ranking, embed_fn=embed_fn,
            entities=entities, entity_freq=entity_freq, caep_gate=caep_gate,
            lqp_model=lqp_model, faiss_search_fn=faiss_search_fn, confidence_fn=confidence_proxy,
            lag_model=fold_lag_models[qid], train=False,
        )

    # 6. Measure Latency (3 Repeats)
    n_repeats = 3
    print(f"Running latency measurement ({n_repeats} repeats)...")
    raw_latencies = []
    v1_latencies = []
    v2_latencies = []
    
    for rep in range(n_repeats):
        print(f"  Repeat {rep + 1}/{n_repeats}...")
        for i, q in enumerate(queries):
            qid, query_text = q["query_id"], q["text"]
            
            # --- RAW ---
            t0_raw = time.perf_counter()
            query_emb = embed_fn([query_text])[0]
            raw_ranking = faiss_search_fn(query_emb)
            raw_time = time.perf_counter() - t0_raw
            raw_latencies.append(raw_time)
            
            # --- v1 ---
            t0_v1 = time.perf_counter()
            setu_v1_fixed_order(
                query=query_text, raw_ranking=raw_ranking, embed_fn=embed_fn,
                entities=entities, entity_freq=entity_freq, caep_gate=caep_gate,
                lqp_model=lqp_model, faiss_search_fn=faiss_search_fn, lag_model=fold_lag_models[qid],
            )
            v1_latencies.append(raw_time + (time.perf_counter() - t0_v1))
            
            # --- v2 ---
            t0_v2 = time.perf_counter()
            setu_v2_run(
                query=query_text, query_id=qid, controller=fold_controllers[qid], raw_ranking=raw_ranking, embed_fn=embed_fn,
                entities=entities, entity_freq=entity_freq, caep_gate=caep_gate,
                lqp_model=lqp_model, faiss_search_fn=faiss_search_fn, confidence_fn=confidence_proxy,
                lag_model=fold_lag_models[qid], train=False,
            )
            v2_latencies.append(raw_time + (time.perf_counter() - t0_v2))

    # 7. Aggregate
    output = {
        "hardware_environment": get_hardware_info(),
        "n_queries": len(queries),
        "n_repeats": f"{n_repeats} (local CPU constraint)",
        "results": {
            "RAW": compute_stats(raw_latencies),
            "SETU_v1": compute_stats(v1_latencies),
            "SETU_v2": compute_stats(v2_latencies)
        }
    }
    
    out_path = root / "results" / "tables" / "latency_final.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4)
        
    print(f"\nSaved latency report to {out_path}")
    print(json.dumps(output["results"], indent=2))

if __name__ == "__main__":
    main()
