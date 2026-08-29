"""
Phase 2 Standalone Operator Ablation:
Evaluates LQP alone, CAEP alone, and LAG alone in isolation against the raw-query
retrieval baseline across all 75 pilot queries (and the Q61-Q75 misspelled subset),
using ranx.evaluate().

Saves results to results/tables/operator_ablation.json and prints summary tables.
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

root = Path(__file__).resolve().parents[1]
sys.path.append(str(root))

from setu.diagnosis.cmi import cmi
from setu.diagnosis.lid_entropy import lid_entropy
from setu.operators.caep import extract_entity_list, entity_frequencies, apply_caep
from setu.operators.lqp import apply_lqp
from setu.operators.lag import entity_density, predict_strategy, apply_lag
from setu.fusion.carf import rrf_baseline


def main():
    tables_dir = root / "results" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load real pilot corpus + queries
    chunks = []
    with open(root / "data" / "processed" / "corpus_chunks.jsonl", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    doc_ids = [c["chunk_id"] for c in chunks]
    doc_texts = [c["text"] for c in chunks]
    queries = json.load(open(root / "data" / "processed" / "queries_remapped.json", encoding="utf-8"))

    # 2. Embedding + FAISS setup (BGE-M3)
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

    # 3. Entities + fitted models
    entities = extract_entity_list(doc_texts)
    entity_freq = entity_frequencies(doc_texts)

    print("Loading fitted models from results/models/...")
    with open(root / "results" / "models" / "caep_gate.pkl", "rb") as f:
        caep_gate = pickle.load(f)
    with open(root / "results" / "models" / "lqp_model.pkl", "rb") as f:
        lqp_model = pickle.load(f)
    with open(root / "results" / "models" / "lag_model.pkl", "rb") as f:
        lag_model = pickle.load(f)

    # 4. Evaluation runs across all queries
    qrels_dict = {q["query_id"]: {d: 1 for d in q["relevant_doc_ids"]} for q in queries}

    raw_run = {}
    lqp_run = {}
    caep_run = {}
    lag_run = {}

    lqp_latencies, caep_latencies, lag_latencies, raw_latencies = [], [], [], []

    print(f"Evaluating {len(queries)} queries across standalone operators...")
    for i, q in enumerate(queries):
        qid, query_text = q["query_id"], q["text"]

        # --- RAW BASELINE ---
        t0 = time.perf_counter()
        query_emb = embed_fn([query_text])[0]
        raw_ids, raw_scores = faiss_search_fn(query_emb)
        raw_latencies.append(time.perf_counter() - t0)
        raw_run[qid] = {doc: (len(raw_ids) - rank) for rank, doc in enumerate(raw_ids)}

        # --- LQP ALONE ---
        t0 = time.perf_counter()
        q_cmi = cmi(query_text)
        projected_emb = apply_lqp(np.asarray(query_emb), q_cmi, lqp_model, cmi_max=1.0)
        lqp_ids, _ = faiss_search_fn(projected_emb)
        lqp_latencies.append(time.perf_counter() - t0)
        lqp_run[qid] = {doc: (len(lqp_ids) - rank) for rank, doc in enumerate(lqp_ids)}

        # --- CAEP ALONE ---
        t0 = time.perf_counter()
        corrected_query = apply_caep(query_text, entities, caep_gate, entity_freq, embed_fn=embed_fn)
        corrected_emb = embed_fn([corrected_query])[0]
        caep_ids, _ = faiss_search_fn(corrected_emb)
        caep_latencies.append(time.perf_counter() - t0)
        caep_run[qid] = {doc: (len(caep_ids) - rank) for rank, doc in enumerate(caep_ids)}

        # --- LAG ALONE ---
        t0 = time.perf_counter()
        q_entropy = lid_entropy(query_text)
        q_density = entity_density(query_text, entities)
        lag_strat = predict_strategy(q_cmi, q_entropy, q_density, lag_model)
        lag_out = apply_lag(
            query_text,
            lag_strat,
            entities=entities,
            embed_fn=embed_fn,
            caep_gate=caep_gate,
            entity_freq=entity_freq,
        )
        if isinstance(lag_out, list):
            q1_emb = embed_fn([lag_out[0]])[0]
            q2_emb = embed_fn([lag_out[1]])[0]
            r1_ids, _ = faiss_search_fn(np.asarray(q1_emb))
            r2_ids, _ = faiss_search_fn(np.asarray(q2_emb))
            lag_ids = rrf_baseline([r1_ids, r2_ids])
        else:
            lag_emb = embed_fn([lag_out])[0]
            lag_ids, _ = faiss_search_fn(np.asarray(lag_emb))
        lag_latencies.append(time.perf_counter() - t0)
        lag_run[qid] = {doc: (len(lag_ids) - rank) for rank, doc in enumerate(lag_ids)}

    METRICS = ["ndcg@10", "mrr", "recall@5", "recall@10"]
    qrels = Qrels(qrels_dict)

    # 5. Evaluate Full Dataset
    results_full = {
        "RAW": {k: float(v) for k, v in evaluate(qrels, Run(raw_run), METRICS).items()},
        "LQP_alone": {k: float(v) for k, v in evaluate(qrels, Run(lqp_run), METRICS).items()},
        "CAEP_alone": {k: float(v) for k, v in evaluate(qrels, Run(caep_run), METRICS).items()},
        "LAG_alone": {k: float(v) for k, v in evaluate(qrels, Run(lag_run), METRICS).items()},
    }
    results_full["RAW"]["mean_latency_ms"] = float(np.mean(raw_latencies) * 1000)
    results_full["LQP_alone"]["mean_latency_ms"] = float(np.mean(lqp_latencies) * 1000)
    results_full["CAEP_alone"]["mean_latency_ms"] = float(np.mean(caep_latencies) * 1000)
    results_full["LAG_alone"]["mean_latency_ms"] = float(np.mean(lag_latencies) * 1000)

    # 6. Evaluate Misspelled Entity Subset (Q61-Q75)
    subset_qids = [q["query_id"] for q in queries if int(q["query_id"].replace("Q", "")) >= 61]
    results_misspelled = {}
    if subset_qids:
        qrels_sub = Qrels({qid: qrels_dict[qid] for qid in subset_qids})
        results_misspelled = {
            "RAW": {k: float(v) for k, v in evaluate(qrels_sub, Run({qid: raw_run[qid] for qid in subset_qids}), METRICS).items()},
            "LQP_alone": {k: float(v) for k, v in evaluate(qrels_sub, Run({qid: lqp_run[qid] for qid in subset_qids}), METRICS).items()},
            "CAEP_alone": {k: float(v) for k, v in evaluate(qrels_sub, Run({qid: caep_run[qid] for qid in subset_qids}), METRICS).items()},
            "LAG_alone": {k: float(v) for k, v in evaluate(qrels_sub, Run({qid: lag_run[qid] for qid in subset_qids}), METRICS).items()},
        }

    output_payload = {
        "full_dataset": results_full,
        "misspelled_subset": results_misspelled,
        "n_total_queries": len(queries),
        "n_misspelled_queries": len(subset_qids),
    }

    out_path = tables_dir / "operator_ablation.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)
    print(f"\nSaved standalone operator ablation table to {out_path}")

    # 7. Print formatted summary
    print("\n" + "=" * 60)
    print(f"=== STANDALONE OPERATOR ABLATION (Full Dataset: {len(queries)} queries) ===")
    print("=" * 60)
    for model_name, metrics in results_full.items():
        print(f"--- {model_name} ---")
        for m, val in metrics.items():
            print(f"  {m:18s}: {val:.4f}")

    if results_misspelled:
        print("\n" + "=" * 60)
        print(f"=== STANDALONE OPERATOR ABLATION (Misspelled Subset: {len(subset_qids)} queries) ===")
        print("=" * 60)
        for model_name, metrics in results_misspelled.items():
            print(f"--- {model_name} ---")
            for m, val in metrics.items():
                print(f"  {m:18s}: {val:.4f}")


if __name__ == "__main__":
    main()
