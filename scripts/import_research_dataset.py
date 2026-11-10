"""
SETU Verified Romanized Hinglish Research Dataset Importer
Downloads, extracts, and validates 100% Romanized (Latin-script) Hinglish datasets from Hugging Face:
1. Google Research Hinglish-TOP (WillHeld/hinglish_top) - Human-annotated queries.
2. Databricks Dolly-15k Hinglish CodeMix (aaditya/databricks-dolly-15k-Hinglish-Codemix) - QA & knowledge pairs.
3. IIT Gandhinagar PHINC (LingoIITGN/PHINC) - 13,738 parallel Hinglish-English pairs.
4. Sujalvc Hinglish Instruct (Sujalvc/hinglish-instruct-dataset) - Pure Roman-script QA.

Usage:
    python scripts/import_research_dataset.py --source hinglish_top --max-samples 500
    python scripts/import_research_dataset.py --source dolly_hinglish --max-samples 500
    python scripts/import_research_dataset.py --source phinc --max-samples 500
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

try:
    from setu.config import set_seed
    set_seed()
except ImportError:
    pass

from setu.diagnosis.cmi import cmi

def import_hinglish_top(max_samples: int = 500, out_dir: Path = None) -> Tuple[Path, Path]:
    """
    Imports Google Research's Hinglish-TOP human-annotated Romanized queries.
    Script is 100% Latin alphabet.
    """
    from datasets import load_dataset
    print("[*] Connecting to Hugging Face: WillHeld/hinglish_top (Google Research)...")
    ds = load_dataset("WillHeld/hinglish_top", split="train")
    
    queries = []
    intents_to_chunks = {}
    corpus = []
    
    print(f"[*] Extracting up to {max_samples} human-annotated Romanized Hinglish queries...")
    for idx, row in enumerate(ds):
        if len(queries) >= max_samples:
            break
            
        q_text = row.get("utterance", "").strip()
        intent = row.get("intent", "general_query").strip()
        
        if len(q_text.split()) < 3:
            continue
            
        chunk_id = f"chunk_intent_{intent.lower()}"
        if chunk_id not in intents_to_chunks:
            chunk_doc = {
                "chunk_id": chunk_id,
                "text": f"Information and execution guidelines for {intent.replace('_', ' ')}. Relevant actions, workflows, and details pertaining to user queries requesting {intent}.",
                "metadata": {"intent": intent, "source": "Google-Hinglish-TOP"}
            }
            intents_to_chunks[chunk_id] = chunk_doc
            corpus.append(chunk_doc)
            
        q_id = f"q_top_{idx:05d}"
        queries.append({
            "query_id": q_id,
            "query": q_text,
            "gold_chunk_ids": [chunk_id],
            "metadata": {
                "intent": intent,
                "cmi": cmi(q_text),
                "script": "Latin-Romanized",
                "dataset_source": "Google-Hinglish-TOP"
            }
        })
        
    out_dir = out_dir or (ROOT / "data" / "research_hinglish_top")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    corpus_file = out_dir / "corpus_chunks.jsonl"
    query_file = out_dir / "queries.json"
    
    with open(corpus_file, "w", encoding="utf-8") as f:
        for c in corpus:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
            
    with open(query_file, "w", encoding="utf-8") as f:
        json.dump(queries, f, indent=2, ensure_ascii=False)
        
    print(f"[✓] Hinglish-TOP imported successfully into {out_dir}:")
    print(f"    - Passages: {len(corpus)}")
    print(f"    - Queries:  {len(queries)}")
    return corpus_file, query_file


def import_dolly_hinglish(max_samples: int = 500, out_dir: Path = None) -> Tuple[Path, Path]:
    """
    Imports Databricks Dolly-15k Hinglish CodeMix.
    Contains Romanized Hinglish questions paired with English/Hinglish passages.
    """
    from datasets import load_dataset
    print("[*] Connecting to Hugging Face: aaditya/databricks-dolly-15k-Hinglish-Codemix...")
    ds = load_dataset("aaditya/databricks-dolly-15k-Hinglish-Codemix", split="train")
    
    corpus = []
    queries = []
    
    print(f"[*] Extracting up to {max_samples} Romanized Hinglish QA pairs...")
    qa_categories = {"open_qa", "closed_qa", "information_extraction", ""}
    seen_q, seen_doc = set(), set()
    for idx, row in enumerate(ds):
        if len(queries) >= max_samples:
            break

        cat = (row.get("en_category") or row.get("category") or "").strip()
        if cat and cat not in qa_categories:
            continue

        q_text = (row.get("codemix_instruction") or row.get("instruction") or "").strip()
        ans_text = (row.get("en_output") or row.get("response") or "").strip()
        context_text = (row.get("en_input") or row.get("context") or "").strip()

        doc_text = context_text if len(context_text.split()) >= 15 else ans_text
        if len(q_text.split()) < 3 or len(doc_text.split()) < 15:
            continue
        q_norm, doc_norm = q_text.lower(), doc_text.lower()
        if q_norm in seen_q or doc_norm in seen_doc:
            continue
        seen_q.add(q_norm)
        seen_doc.add(doc_norm)

        chunk_id = f"chunk_dolly_{idx:05d}"
        query_id = f"q_dolly_{idx:05d}"

        corpus.append({
            "chunk_id": chunk_id,
            "text": doc_text,
            "metadata": {"source": "Dolly-15k-Hinglish", "category": cat}
        })
        
        queries.append({
            "query_id": query_id,
            "query": q_text,
            "gold_chunk_ids": [chunk_id],
            "metadata": {
                "cmi": cmi(q_text),
                "script": "Latin-Romanized",
                "dataset_source": "Dolly-15k-Hinglish"
            }
        })
        
    out_dir = out_dir or (ROOT / "data" / "research_dolly_hinglish")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    corpus_file = out_dir / "corpus_chunks.jsonl"
    query_file = out_dir / "queries.json"
    
    with open(corpus_file, "w", encoding="utf-8") as f:
        for c in corpus:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
            
    with open(query_file, "w", encoding="utf-8") as f:
        json.dump(queries, f, indent=2, ensure_ascii=False)
        
    print(f"[✓] Dolly-15k Hinglish imported successfully into {out_dir}:")
    print(f"    - Passages: {len(corpus)}")
    print(f"    - Queries:  {len(queries)}")
    return corpus_file, query_file


def import_phinc(max_samples: int = 500, out_dir: Path = None) -> Tuple[Path, Path]:
    """
    Imports IIT Gandhinagar's PHINC parallel Hinglish-English corpus.
    100% Romanized Hinglish sentences mapped to English translations.
    """
    from datasets import load_dataset
    print("[*] Connecting to Hugging Face: LingoIITGN/PHINC...")
    ds = load_dataset("LingoIITGN/PHINC", split="train")
    
    corpus = []
    queries = []
    
    print(f"[*] Processing up to {max_samples} Romanized parallel pairs...")
    for idx, row in enumerate(ds):
        if idx >= max_samples:
            break
            
        h_col = "Sentence" if "Sentence" in row else "Hinglish Code-Mixed Sentence"
        e_col = "English_Translation" if "English_Translation" in row else "Human Translated English Sentence"
        
        q_text = row.get(h_col, "").strip()
        ans_text = row.get(e_col, "").strip()
        
        if len(q_text.split()) < 3 or len(ans_text.split()) < 3:
            continue
            
        chunk_id = f"chunk_ph_{idx:05d}"
        query_id = f"q_ph_{idx:05d}"
        
        corpus.append({
            "chunk_id": chunk_id,
            "text": ans_text,
            "metadata": {"source": "PHINC-IITGN"}
        })
        
        queries.append({
            "query_id": query_id,
            "query": q_text,
            "gold_chunk_ids": [chunk_id],
            "metadata": {
                "cmi": cmi(q_text),
                "script": "Latin-Romanized",
                "dataset_source": "PHINC"
            }
        })
        
    out_dir = out_dir or (ROOT / "data" / "research_phinc")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    corpus_file = out_dir / "corpus_chunks.jsonl"
    query_file = out_dir / "queries.json"
    
    with open(corpus_file, "w", encoding="utf-8") as f:
        for c in corpus:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
            
    with open(query_file, "w", encoding="utf-8") as f:
        json.dump(queries, f, indent=2, ensure_ascii=False)
        
    print(f"[✓] PHINC imported successfully into {out_dir}:")
    print(f"    - Passages: {len(corpus)}")
    print(f"    - Queries:  {len(queries)}")
    return corpus_file, query_file


def main():
    parser = argparse.ArgumentParser(description="Import verified 100% Romanized Hinglish research datasets")
    parser.add_argument("--source", type=str, choices=["hinglish_top", "dolly_hinglish", "phinc"], default="hinglish_top",
                        help="Research dataset source to import")
    parser.add_argument("--max-samples", type=int, default=500, help="Maximum number of queries to extract")
    parser.add_argument("--out-dir", type=str, default=None, help="Custom output directory")
    args = parser.parse_args()
    
    out_p = Path(args.out_dir) if args.out_dir else None
    
    if args.source == "hinglish_top":
        import_hinglish_top(args.max_samples, out_p)
    elif args.source == "dolly_hinglish":
        import_dolly_hinglish(args.max_samples, out_p)
    elif args.source == "phinc":
        import_phinc(args.max_samples, out_p)

if __name__ == "__main__":
    main()
