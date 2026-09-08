"""
Sync queries from data/pilot_corpus/queries.json to data/processed/queries_remapped.json.

Maintains atomic D01-D20 document IDs matching the atomic corpus chunks
in data/processed/corpus_chunks.jsonl.
"""
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config

queries = json.load(open(config.DATA_PILOT / "queries.json", encoding="utf-8"))

out_path = Path("data/processed/queries_remapped.json")
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(queries, f, indent=2, ensure_ascii=False)

print(f"Saved {len(queries)} atomic queries -> {out_path}")

# Quick sanity check: show first 3
for q in queries[:3]:
    print(q["query_id"], "->", q["relevant_doc_ids"], ":", q["text"][:50], "...")