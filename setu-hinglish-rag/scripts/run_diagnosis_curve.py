"""
Phase 1 deliverable: End-to-end CMI-vs-Recall/MRR/nDCG diagnosis curve for all 3 embedding models
(BGE-M3, Indic-SBERT, mE5-Large) on the 20 atomic corpus chunks.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import faiss
import matplotlib.pyplot as plt
import numpy as np
from ranx import Qrels, Run, evaluate
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import setu.config as config
from setu.diagnosis.cmi import cmi

LOG_DIR = ROOT / "results" / "logs"
FIG_DIR = ROOT / "results" / "figures"
LOG_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

MODELS = {
    "indic_sbert": ("l3cube-pune/indic-sentence-similarity-sbert", False),
    "bge_m3": ("BAAI/bge-m3", False),
    "me5_large": ("intfloat/multilingual-e5-large", True),
}


def cmi_band(score, bands):
    for lo, hi, name in bands:
        if lo <= score < hi:
            return name
    return bands[-1][2]


def main():
    # 1. Load corpus and queries
    corpus_path = ROOT / "data" / "processed" / "corpus_chunks.jsonl"
    queries_path = ROOT / "data" / "processed" / "queries_remapped.json"

    chunks = []
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))

    queries = json.load(open(queries_path, "r", encoding="utf-8"))

    doc_ids = [c["chunk_id"] for c in chunks]
    doc_texts = [c["text"] for c in chunks]
    query_ids = [q["query_id"] for q in queries]
    query_texts = [q["text"] for q in queries]

    print(f"Loaded {len(chunks)} corpus chunks and {len(queries)} queries.")

    # Build Qrels
    qrels_dict = {q["query_id"]: {d: 1 for d in q["relevant_doc_ids"]} for q in queries}
    qrels = Qrels(qrels_dict)

    # Compute CMI & bands
    query_bands = {}
    for q in queries:
        c = cmi(q["text"])
        b = cmi_band(c, config.CMI_BANDS)
        query_bands[q["query_id"]] = b

    METRICS = ["ndcg@10", "mrr", "recall@5", "recall@10", "hit_rate@1"]
    all_retrieval_results = {}
    per_query_metrics = {}

    for model_key, (model_name, needs_prefix) in MODELS.items():
        print(f"\n==================================================")
        print(f"Embedding & Retrieving: {model_key} ({model_name})")
        print(f"==================================================")
        model = SentenceTransformer(model_name)

        q_input = [f"query: {t}" for t in query_texts] if needs_prefix else query_texts
        d_input = [f"passage: {t}" for t in doc_texts] if needs_prefix else doc_texts

        q_emb = model.encode(q_input, show_progress_bar=False, convert_to_numpy=True).astype("float32")
        d_emb = model.encode(d_input, show_progress_bar=False, convert_to_numpy=True).astype("float32")

        faiss.normalize_L2(q_emb)
        faiss.normalize_L2(d_emb)

        np.save(LOG_DIR / f"query_emb_{model_key}.npy", q_emb)
        np.save(LOG_DIR / f"doc_emb_{model_key}.npy", d_emb)

        index = faiss.IndexFlatIP(d_emb.shape[1])
        index.add(d_emb)

        top_k = config.TOP_K
        scores, indices = index.search(q_emb, top_k)

        run_dict = {}
        model_results = {}
        for i, qid in enumerate(query_ids):
            ranked_doc_ids = [doc_ids[idx] for idx in indices[i]]
            ranked_scores = [float(s) for s in scores[i]]
            run_dict[qid] = {d: s for d, s in zip(ranked_doc_ids, ranked_scores)}
            model_results[qid] = {
                "ranked_doc_ids": ranked_doc_ids,
                "scores": ranked_scores,
            }

        all_retrieval_results[model_key] = model_results
        run = Run(run_dict)

        overall = evaluate(qrels, run, METRICS)
        print(f"Overall Metrics ({model_key}):", overall)

        per_q = evaluate(qrels, run, METRICS, return_mean=False)
        per_query_metrics[model_key] = {
            qid: {m: float(per_q[m][i]) for m in METRICS}
            for i, qid in enumerate(query_ids)
        }

    # Save log artifacts
    with open(LOG_DIR / "query_ids.json", "w", encoding="utf-8") as f:
        json.dump(query_ids, f, indent=2)
    with open(LOG_DIR / "doc_ids.json", "w", encoding="utf-8") as f:
        json.dump(doc_ids, f, indent=2)
    with open(LOG_DIR / "retrieval_results.json", "w", encoding="utf-8") as f:
        json.dump(all_retrieval_results, f, indent=2)
    with open(LOG_DIR / "per_query_metrics.json", "w", encoding="utf-8") as f:
        json.dump(per_query_metrics, f, indent=2)
    print("\nUpdated all JSON log files in results/logs/")

    # 4. Print Per-Band Summary Table for All Models
    BAND_ORDER = [b[2] for b in config.CMI_BANDS]
    print("\n" + "=" * 80)
    print("PER-CMI-BAND RETRIEVAL DEGRADATION TABLE (20-CHUNK ATOMIC CORPUS)")
    print("=" * 80)
    for model_key in MODELS:
        print(f"\nModel: {model_key}")
        print(f"{'Band':<12} {'Count':<8} {'MRR':<10} {'nDCG@10':<12} {'Recall@5':<10} {'Hit@1':<10}")
        print("-" * 62)
        for band in BAND_ORDER:
            q_in_band = [qid for qid, b in query_bands.items() if b == band]
            if not q_in_band:
                continue
            mrrs = [per_query_metrics[model_key][qid]["mrr"] for qid in q_in_band]
            ndcgs = [per_query_metrics[model_key][qid]["ndcg@10"] for qid in q_in_band]
            rec5s = [per_query_metrics[model_key][qid]["recall@5"] for qid in q_in_band]
            hit1s = [per_query_metrics[model_key][qid]["hit_rate@1"] for qid in q_in_band]

            print(f"{band:<12} {len(q_in_band):<8} {np.mean(mrrs):<10.4f} {np.mean(ndcgs):<12.4f} {np.mean(rec5s):<10.4f} {np.mean(hit1s):<10.4f}")

    # 5. Plot and save degradation curve figure
    plt.figure(figsize=(9, 5.5))
    for model_key in MODELS:
        x = []
        y = []
        for band in BAND_ORDER:
            q_in_band = [qid for qid, b in query_bands.items() if b == band]
            if q_in_band:
                ndcgs = [per_query_metrics[model_key][qid]["ndcg@10"] for qid in q_in_band]
                x.append(f"{band}\n(n={len(q_in_band)})")
                y.append(np.mean(ndcgs))
        plt.plot(x, y, marker="o", linewidth=2, label=model_key)

    plt.xlabel("Code-Mixing Index (CMI) Band", fontsize=11)
    plt.ylabel("Average nDCG@10", fontsize=11)
    plt.title("Retrieval Degradation vs. Code-Mixing Index (20-Chunk Atomic Pilot)", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out_fig = FIG_DIR / "degradation_curve.png"
    plt.savefig(out_fig, dpi=150)
    print(f"\nSaved updated degradation curve plot to: {out_fig}")


if __name__ == "__main__":
    main()
