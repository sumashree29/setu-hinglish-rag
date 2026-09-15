import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
try:
    from setu.config import set_seed
    set_seed()
except ImportError:
    pass

import json
import re
import numpy as np
from collections import Counter
import setu.config as config
from setu.diagnosis.cmi import cmi
from setu.diagnosis.lid_entropy import lid_entropy

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results" / "tables"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def cmi_band(score, bands):
    for lo, hi, name in bands:
        if lo <= score < hi:
            return name
    return bands[-1][2]

def main():
    # Load chunks
    chunks_file = DATA_DIR / "processed" / "corpus_chunks_v2.jsonl"
    chunks = [json.loads(line) for line in open(chunks_file, encoding='utf-8') if line.strip()]
    
    source_docs = set(c.get("doc_id", c.get("source_id", "unknown")) for c in chunks)
    num_chunks = len(chunks)
    
    chunk_lengths = [len(re.findall(r"\b\w+\b", c["text"])) for c in chunks]
    len_mean = np.mean(chunk_lengths)
    len_median = np.median(chunk_lengths)
    len_min = np.min(chunk_lengths)
    len_max = np.max(chunk_lengths)
    
    # Load queries
    queries_file = DATA_DIR / "processed" / "queries_v3_final.json"
    queries = json.load(open(queries_file, encoding='utf-8'))
    
    num_queries_by_variant = Counter(q.get("variant", "unknown") for q in queries)
    num_queries_by_review_status = Counter(q.get("review_status", "unreviewed") for q in queries)
    
    cmis = [cmi(q["text"]) for q in queries]
    entropies = [lid_entropy(q["text"]) for q in queries]
    
    cmi_bands_counts = Counter(cmi_band(c, config.CMI_BANDS) for c in cmis)
    
    rel_docs_per_query = [len(q.get("relevant_doc_ids", [])) for q in queries]
    queries_with_multi_rel = sum(1 for count in rel_docs_per_query if count > 1)
    
    stats = {
        "num_source_documents": len(source_docs),
        "num_chunks": num_chunks,
        "chunk_length_distribution": {
            "mean": float(len_mean),
            "median": float(len_median),
            "min": int(len_min),
            "max": int(len_max)
        },
        "num_queries_by_variant": dict(num_queries_by_variant),
        "num_queries_by_review_status": dict(num_queries_by_review_status),
        "cmi_distribution": {
            "mean": float(np.mean(cmis)),
            "median": float(np.median(cmis)),
            "min": float(np.min(cmis)),
            "max": float(np.max(cmis))
        },
        "cmi_band_counts": dict(cmi_bands_counts),
        "lid_entropy_distribution": {
            "mean": float(np.mean(entropies)),
            "median": float(np.median(entropies)),
            "min": float(np.min(entropies)),
            "max": float(np.max(entropies))
        },
        "num_relevant_docs_per_query": {
            "mean": float(np.mean(rel_docs_per_query)),
            "median": float(np.median(rel_docs_per_query)),
            "min": int(np.min(rel_docs_per_query)),
            "max": int(np.max(rel_docs_per_query))
        },
        "num_queries_with_more_than_one_relevant_doc": queries_with_multi_rel,
        "total_queries": len(queries)
    }
    
    json_path = RESULTS_DIR / "dataset_statistics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
        
    # Write Markdown
    md_path = RESULTS_DIR / "dataset_statistics.md"
    md_content = f"""# Dataset Statistics

## Document Statistics
- **Source Documents**: {stats['num_source_documents']}
- **Total Chunks**: {stats['num_chunks']}

### Chunk Length Distribution (Words)
- **Mean**: {stats['chunk_length_distribution']['mean']:.2f}
- **Median**: {stats['chunk_length_distribution']['median']}
- **Min**: {stats['chunk_length_distribution']['min']}
- **Max**: {stats['chunk_length_distribution']['max']}

## Query Statistics
- **Total Queries**: {stats['total_queries']}

### Queries by Variant
"""
    for k, v in stats['num_queries_by_variant'].items():
        md_content += f"- {k}: {v}\n"
        
    md_content += f"\n### Queries by Review Status\n"
    for k, v in stats['num_queries_by_review_status'].items():
        md_content += f"- {k}: {v}\n"
        
    md_content += f"\n### CMI Distribution\n"
    md_content += f"- **Mean**: {stats['cmi_distribution']['mean']:.3f}\n"
    md_content += f"- **Median**: {stats['cmi_distribution']['median']:.3f}\n"
    md_content += f"- **Min**: {stats['cmi_distribution']['min']:.3f}\n"
    md_content += f"- **Max**: {stats['cmi_distribution']['max']:.3f}\n"
    
    md_content += f"\n### CMI Band Counts\n"
    for k, v in stats['cmi_band_counts'].items():
        md_content += f"- {k}: {v}\n"

    md_content += f"\n### LID Entropy Distribution\n"
    md_content += f"- **Mean**: {stats['lid_entropy_distribution']['mean']:.3f}\n"
    md_content += f"- **Median**: {stats['lid_entropy_distribution']['median']:.3f}\n"
    md_content += f"- **Min**: {stats['lid_entropy_distribution']['min']:.3f}\n"
    md_content += f"- **Max**: {stats['lid_entropy_distribution']['max']:.3f}\n"
    
    md_content += f"\n### Relevance Labels\n"
    md_content += f"- **Mean relevant docs per query**: {stats['num_relevant_docs_per_query']['mean']:.2f}\n"
    md_content += f"- **Queries with >1 relevant doc**: {stats['num_queries_with_more_than_one_relevant_doc']}\n"
    
    md_content += f"""
## Relevance-Label Construction Procedure
Each query has exactly one gold chunk inherited from its `source_question`'s parent chunk.
Labels are binary, not graded. No pooling was performed.
**Consequence:** `Recall@10 = 0.987` is near-saturation, so Recall is uninformative on this corpus and **MRR / nDCG@10 / Hit@1 must be the primary metrics**.
"""
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
        
    print(f"Saved dataset stats to {json_path} and {md_path}")

if __name__ == "__main__":
    main()
