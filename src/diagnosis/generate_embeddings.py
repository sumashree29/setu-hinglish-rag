"""
Generate and save embeddings for the full pilot corpus (20 docs, 60 queries)
using all 3 embedding models. Saves to results/logs/ as .npy files so we
don't need to re-run this every time -- embedding is the slow step.
"""
import json
import sys
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config

OUT_DIR = config.ROOT / "results" / "logs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

docs = json.load(open(config.DATA_PILOT / "documents.json", encoding="utf-8"))
queries = json.load(open(config.DATA_PILOT / "queries.json", encoding="utf-8"))

doc_texts = [d["text"] for d in docs]
doc_ids = [d["doc_id"] for d in docs]
query_texts = [q["text"] for q in queries]
query_ids = [q["query_id"] for q in queries]

MODELS = {
    "indic_sbert": ("l3cube-pune/indic-sentence-similarity-sbert", False),
    "bge_m3": ("BAAI/bge-m3", False),
    "me5_large": ("intfloat/multilingual-e5-large", True),
}

for model_key, (model_name, needs_prefix) in MODELS.items():
    print(f"\n=== Embedding with {model_key} ({model_name}) ===")
    model = SentenceTransformer(model_name)

    if needs_prefix:
        q_input = [f"query: {t}" for t in query_texts]
        d_input = [f"passage: {t}" for t in doc_texts]
    else:
        q_input = query_texts
        d_input = doc_texts

    print("Encoding queries...")
    q_emb = model.encode(q_input, show_progress_bar=True, convert_to_numpy=True)
    print("Encoding docs...")
    d_emb = model.encode(d_input, show_progress_bar=True, convert_to_numpy=True)

    np.save(OUT_DIR / f"query_emb_{model_key}.npy", q_emb)
    np.save(OUT_DIR / f"doc_emb_{model_key}.npy", d_emb)
    print(f"Saved: query_emb_{model_key}.npy {q_emb.shape}, doc_emb_{model_key}.npy {d_emb.shape}")

# Save the id orderings too, so we know which row = which query/doc later
with open(OUT_DIR / "query_ids.json", "w") as f:
    json.dump(query_ids, f)
with open(OUT_DIR / "doc_ids.json", "w") as f:
    json.dump(doc_ids, f)

print("\nAll embeddings generated and saved to", OUT_DIR)