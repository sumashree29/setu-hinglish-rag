"""
Join per-query CMI scores with per-query retrieval metrics, bin by CMI band,
and plot the degradation curve -- the core Phase 1 deliverable proving H1/H2.
"""
import json
import sys
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parents[1]))
sys.path.append(str(Path(__file__).resolve().parent))
import config
from lid import tag_tokens
from cmi import compute_cmi, cmi_band

LOG_DIR = config.ROOT / "results" / "logs"
FIG_DIR = config.ROOT / "results" / "figures"

queries = json.load(open(config.DATA_PILOT / "queries.json", encoding="utf-8"))
per_query_metrics = json.load(open(LOG_DIR / "per_query_metrics.json"))

# Compute CMI + band for every query
query_cmi = {}
query_band = {}
for q in queries:
    tags = tag_tokens(q["text"])
    c = compute_cmi(tags)
    b = cmi_band(c, config.CMI_BANDS)
    query_cmi[q["query_id"]] = c
    query_band[q["query_id"]] = b

MODEL_KEYS = ["indic_sbert", "bge_m3", "me5_large"]
METRIC_TO_PLOT = "ndcg@10"
BAND_ORDER = [b[2] for b in config.CMI_BANDS]  # ["low", "medium", "high", "very_high"]

# For each model, compute average metric per CMI band
model_band_avg = {}  # {model_key: {band: avg_metric}}
for model_key in MODEL_KEYS:
    band_values = defaultdict(list)
    for qid, metrics in per_query_metrics[model_key].items():
        band = query_band[qid]
        band_values[band].append(metrics[METRIC_TO_PLOT])

    model_band_avg[model_key] = {
        band: (sum(vals) / len(vals) if vals else None)
        for band, vals in band_values.items()
    }
    print(f"\n{model_key} — avg {METRIC_TO_PLOT} by CMI band:")
    for band in BAND_ORDER:
        val = model_band_avg[model_key].get(band)
        n = len(band_values.get(band, []))
        print(f"  {band:12s} (n={n}): {val}")

# Plot
plt.figure(figsize=(8, 5))
for model_key in MODEL_KEYS:
    x = []
    y = []
    for band in BAND_ORDER:
        val = model_band_avg[model_key].get(band)
        if val is not None:
            x.append(band)
            y.append(val)
    plt.plot(x, y, marker="o", label=model_key)

plt.xlabel("CMI Band (Code-Mixing Index)")
plt.ylabel(f"Average {METRIC_TO_PLOT}")
plt.title("Retrieval Degradation vs. Code-Mixing Index (Phase 1 Pilot)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

FIG_DIR.mkdir(parents=True, exist_ok=True)
out_path = FIG_DIR / "degradation_curve.png"
plt.savefig(out_path, dpi=150)
print(f"\nSaved plot to {out_path}")