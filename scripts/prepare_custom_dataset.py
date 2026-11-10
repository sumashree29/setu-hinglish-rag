"""
SETU Custom Dataset Ingestion & Validation Pipeline
Allows swapping in any new custom corpus and query set into the SETU Hinglish RAG benchmark.

Usage:
    python scripts/prepare_custom_dataset.py --corpus path/to/corpus.jsonl --queries path/to/queries.json --out-dir data/custom/
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Set

def validate_corpus(corpus_path: Path) -> List[Dict]:
    """Validate corpus JSONL format: must have chunk_id and text."""
    chunks = []
    chunk_ids = set()
    print(f"[*] Validating corpus: {corpus_path}")
    
    with open(corpus_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except Exception as e:
                raise ValueError(f"Invalid JSON on line {idx} of {corpus_path}: {e}")
                
            if "chunk_id" not in item:
                raise ValueError(f"Line {idx} missing required key 'chunk_id'")
            if "text" not in item or not item["text"].strip():
                raise ValueError(f"Line {idx} missing or empty 'text'")
                
            cid = str(item["chunk_id"])
            if cid in chunk_ids:
                raise ValueError(f"Duplicate chunk_id '{cid}' found at line {idx}")
                
            chunk_ids.add(cid)
            chunks.append({
                "chunk_id": cid,
                "text": item["text"].strip(),
                "metadata": item.get("metadata", {})
            })
            
    print(f"[+] Corpus validated successfully: {len(chunks)} chunks loaded.")
    return chunks


def validate_queries(queries_path: Path, valid_chunk_ids: Set[str]) -> List[Dict]:
    """Validate queries JSON format: must have query_id, query text, and valid gold_chunk_ids."""
    print(f"[*] Validating queries: {queries_path}")
    with open(queries_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    if not isinstance(data, list):
        raise ValueError(f"Query file {queries_path} must be a JSON array of objects")
        
    queries = []
    query_ids = set()
    
    for idx, item in enumerate(data, 1):
        if "query_id" not in item:
            raise ValueError(f"Query item #{idx} missing required key 'query_id'")
        if "query" not in item or not str(item["query"]).strip():
            raise ValueError(f"Query item #{idx} missing or empty 'query'")
        if "gold_chunk_ids" not in item and "relevant_doc_ids" not in item:
            raise ValueError(f"Query item #{idx} missing 'gold_chunk_ids' or 'relevant_doc_ids'")
            
        qid = str(item["query_id"])
        if qid in query_ids:
            raise ValueError(f"Duplicate query_id '{qid}' found at item #{idx}")
        query_ids.add(qid)
        
        gold_ids = [str(x) for x in item.get("gold_chunk_ids", item.get("relevant_doc_ids", []))]
        if not gold_ids:
            raise ValueError(f"Query item #{idx} ('{qid}') has empty gold targets")
            
        # Verify that gold IDs exist in the corpus
        missing = [gid for gid in gold_ids if gid not in valid_chunk_ids]
        if missing:
            print(f"[!] Warning: Query '{qid}' targets chunk IDs not in corpus: {missing}")
            
        queries.append({
            "query_id": qid,
            "query": str(item["query"]).strip(),
            "gold_chunk_ids": gold_ids,
            "metadata": item.get("metadata", {})
        })
        
    print(f"[+] Queries validated successfully: {len(queries)} queries loaded.")
    return queries


def main():
    parser = argparse.ArgumentParser(description="Ingest and validate custom dataset for SETU Hinglish RAG")
    parser.add_argument("--corpus", type=str, required=True, help="Path to custom corpus .jsonl")
    parser.add_argument("--queries", type=str, required=True, help="Path to custom queries .json")
    parser.add_argument("--out-dir", type=str, default="data/custom", help="Output directory")
    args = parser.parse_args()
    
    corpus_p = Path(args.corpus)
    queries_p = Path(args.queries)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if not corpus_p.exists():
        print(f"Error: Corpus file not found: {corpus_p}")
        return
    if not queries_p.exists():
        print(f"Error: Queries file not found: {queries_p}")
        return
        
    chunks = validate_corpus(corpus_p)
    valid_cids = {c["chunk_id"] for c in chunks}
    queries = validate_queries(queries_p, valid_cids)
    
    out_corpus = out_dir / "corpus_chunks.jsonl"
    out_queries = out_dir / "queries.json"
    
    with open(out_corpus, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
            
    with open(out_queries, "w", encoding="utf-8") as f:
        json.dump(queries, f, indent=2, ensure_ascii=False)
        
    print(f"\n[✓] Successfully ingested and formatted dataset into {out_dir}:")
    print(f"    - Corpus:  {out_corpus} ({len(chunks)} passages)")
    print(f"    - Queries: {out_queries} ({len(queries)} queries)")
    print("\nNext step: Run embedding generation and evaluation on your new dataset:")
    print(f"    python setu/diagnosis/generate_embeddings.py --corpus {out_corpus} --queries {out_queries}")

if __name__ == "__main__":
    main()

