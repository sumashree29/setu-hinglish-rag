"""
Build atomic pilot corpus chunks from data/pilot_corpus/documents.json.

WHY ATOMIC CHUNKING:
At pilot scale, documents.json contains 20 discrete FAQ / scheme entries (D01-D20)
averaging ~33 words each (~656 words total across the entire corpus).
Merging these into 4 large composite chunks (e.g. C01-C04) collapses retrieval discriminability,
making top-1 and top-3 accuracy trivial (or ambiguous) since each chunk contains multiple
unrelated sub-topics.
The plan's 300-500 token chunk guidance applies to Phase 5's full-scale corpus scraped
from raw web circulars, not this 20-entry curated pilot. Keeping D01-D20 atomic preserves
exact 1-to-1 relevance mapping for the 60 benchmark queries.
"""
import json
from pathlib import Path


def build_corpus_chunks():
    root = Path(__file__).resolve().parents[1]
    input_path = root / "data" / "pilot_corpus" / "documents.json"
    output_path = root / "data" / "processed" / "corpus_chunks.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(input_path, "r", encoding="utf-8") as f:
        docs = json.load(f)

    with open(output_path, "w", encoding="utf-8") as f:
        for doc in docs:
            chunk = {
                "chunk_id": doc["doc_id"],
                "text": doc["text"],
            }
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    print(f"Built {len(docs)} atomic corpus chunks -> {output_path}")


if __name__ == "__main__":
    build_corpus_chunks()
