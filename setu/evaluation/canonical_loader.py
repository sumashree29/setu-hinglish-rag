import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
CANONICAL_DIR = ROOT / "results" / "canonical"

def load_canonical_data():
    """
    Loads the canonical results (Phase 8).
    Note: These files are currently PLACEHOLDERS pending the execution of the full
    baseline reruns (Phase 9/10) on a GPU environment.
    """
    data = {}
    
    files = {
        "aggregate_metrics": "aggregate_metrics.json",
        "statistical_tests": "statistical_tests.json",
        "controller_behavior": "controller_behavior.json",
        "latency_metrics": "latency_metrics.json",
        "experiment_metadata": "experiment_metadata.json"
    }
    
    for key, filename in files.items():
        filepath = CANONICAL_DIR / filename
        if filepath.exists():
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data[key] = json.load(f)
            except json.JSONDecodeError:
                data[key] = {}
        else:
            data[key] = {}
            
    # Load jsonl separately
    per_query_path = CANONICAL_DIR / "per_query_results.jsonl"
    per_query = []
    if per_query_path.exists():
        with open(per_query_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip() and line.strip() != "{}":
                    try:
                        per_query.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    data["per_query_results"] = per_query
    
    return data

if __name__ == "__main__":
    canonical_data = load_canonical_data()
    print("Loaded canonical data framework.")
    print("WARNING: Files are placeholders pending Colab execution for Phases 9/10.")
