"""
Compute retrieval metrics (nDCG@10, MRR, Recall@5, Recall@10) per query,
per model, using ranx. Saves per-query metrics so we can later bin them
by CMI band.
"""
import json
import sys
from pathlib import Path

from ranx import Qrels, Run, evaluate

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config

LOG_DIR = config.ROOT / "results" / "logs"
queries = json.load(open("data/processed/queries_remapped.json", encoding="utf-8"))
retrieval_results = json.load(open(LOG_DIR / "retrieval_results.json"))

# Build qrels: query_id -> {doc_id: relevance} (1 = relevant, since we only
# have binary ground truth from our pilot corpus construction)
qrels_dict = {}
for q in queries:
    qrels_dict[q["query_id"]] = {doc_id: 1 for doc_id in q["relevant_doc_ids"]}
qrels = Qrels(qrels_dict)

MODEL_KEYS = ["indic_sbert", "bge_m3", "me5_large"]
METRICS = ["ndcg@10", "mrr", "recall@5", "recall@10"]

per_query_metrics = {}  # {model_key: {query_id: {metric: value}}}

for model_key in MODEL_KEYS:
    print(f"\n=== Metrics for {model_key} ===")
    model_results = retrieval_results[model_key]

    run_dict = {}
    for qid, r in model_results.items():
        run_dict[qid] = {doc_id: score for doc_id, score in zip(r["ranked_doc_ids"], r["scores"])}
    run = Run(run_dict)

    # Overall aggregate metrics
    overall = evaluate(qrels, run, METRICS)
    print("Overall:", overall)

    # Per-query metrics (needed for CMI binning later)
    per_query = evaluate(qrels, run, METRICS, return_mean=False)
    per_query_metrics[model_key] = {
        qid: {m: float(per_query[m][i]) for m in METRICS}
        for i, qid in enumerate(run_dict.keys())
    }

with open(LOG_DIR / "per_query_metrics.json", "w") as f:
    json.dump(per_query_metrics, f, indent=2)

print("\nSaved per-query metrics to", LOG_DIR / "per_query_metrics.json")