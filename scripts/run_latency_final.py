import json
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def get_stats(arr):
    return {
        "mean_ms": float(np.mean(arr) * 1000),
        "median_ms": float(np.median(arr) * 1000),
        "std_ms": float(np.std(arr) * 1000),
        "iqr_ms": float((np.percentile(arr, 75) - np.percentile(arr, 25)) * 1000),
        "p95_ms": float(np.percentile(arr, 95) * 1000)
    }

def main():
    per_query = []
    with open(ROOT / "results" / "canonical" / "per_query_results.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            per_query.append(json.loads(line))
            
    raw_lat = np.array([q["raw_latency"] for q in per_query])
    v1_lat = np.array([q["v1_latency"] for q in per_query])
    v2_lat = np.array([q["v2_latency"] for q in per_query])
    
    out = {
        "protocol": {
            "machine": "Generic CPU inference host",
            "warm_up": "None explicit, evaluated sequentially",
            "included": "Embedding generation, FAISS search, operator execution, controller inference",
            "excluded": "Initial model loading to RAM, file I/O",
        },
        "RAW": get_stats(raw_lat),
        "SETU_v1": get_stats(v1_lat),
        "SETU_v2": get_stats(v2_lat)
    }
    
    with open(ROOT / "results" / "canonical" / "latency_results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=4)
        
    print(f"Latency analysis saved to results/canonical/latency_results.json")

if __name__ == "__main__":
    main()
