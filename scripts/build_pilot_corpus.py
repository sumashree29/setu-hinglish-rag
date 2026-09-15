import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
try:
    from setu.config import set_seed
    set_seed()
except ImportError:
    pass

"""
OWNER: R1 (+ team for query writing) | PHASE: 1
Run after download_models.py. Produces:
  data/processed/corpus_chunks.jsonl
  data/processed/pilot_queries_template.csv  <- team fills this in by hand
"""
from setu.diagnosis.corpus import build_pilot_corpus

if __name__ == "__main__":
    build_pilot_corpus(n_queries=100)
