"""
OWNER: R1 (+ team for query writing) | PHASE: 1
Run after download_models.py. Produces:
  data/processed/corpus_chunks.jsonl
  data/processed/pilot_queries_template.csv  <- team fills this in by hand
"""
from setu.diagnosis.corpus import build_pilot_corpus

if __name__ == "__main__":
    build_pilot_corpus(n_queries=100)
