"""
Phase 3.2 — Confidence-signal correlation study (plan §5.2, H10).
OWNER: R3
"""
import json
from pathlib import Path
from setu.evaluation.metrics import confidence_proxy
from setu.evaluation.stats import spearman_correlation


def run_correlation_study(metrics_path: str, retrieval_path: str) -> dict:
    metrics = json.load(open(metrics_path, encoding="utf-8"))
    retrieval = json.load(open(retrieval_path, encoding="utf-8"))

    results = {}
    for model in metrics:
        margins, entropies, mrrs, recalls = [], [], [], []
        for query_id in metrics[model]:
            scores = retrieval[model][query_id]["scores"]
            margins.append(confidence_proxy(scores, method="margin"))
            entropies.append(confidence_proxy(scores, method="entropy"))
            mrrs.append(metrics[model][query_id]["mrr"])
            recalls.append(metrics[model][query_id]["recall@5"])

        results[model] = {
            "n_queries": len(margins),
            "margin_vs_mrr": spearman_correlation(margins, mrrs),
            "entropy_vs_mrr": spearman_correlation(entropies, mrrs),
            "margin_vs_recall5": spearman_correlation(margins, recalls),
            "entropy_vs_recall5": spearman_correlation(entropies, recalls),
        }
    return results


def print_report(results: dict):
    for model, r in results.items():
        print(f"\n=== {model} (n={r['n_queries']} queries) ===")
        for key in ["margin_vs_mrr", "entropy_vs_mrr", "margin_vs_recall5", "entropy_vs_recall5"]:
            rho, p = r[key]
            sig = "significant" if p < 0.05 else "NOT significant"
            print(f"  {key:22s}: rho={rho:+.3f}, p={p:.4f} ({sig})")


if __name__ == "__main__":
    RESULTS_LOGS = Path("results/logs")
    results = run_correlation_study(
        RESULTS_LOGS / "per_query_metrics.json",
        RESULTS_LOGS / "retrieval_results.json",
    )
    print_report(results)

    output_path = RESULTS_LOGS / "confidence_correlation_study.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {output_path}")
