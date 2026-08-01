"""
Central config — paths and constants shared across every module.
Nobody should hardcode a path anywhere else; import from here instead.
"""
from pathlib import Path

ROOT = Path(__file__).parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_EMBEDDINGS = ROOT / "data" / "embeddings"
DATA_LOGS = ROOT / "data" / "logs"
OUTPUTS_FIGURES = ROOT / "outputs" / "figures"
OUTPUTS_TABLES = ROOT / "outputs" / "tables"

# Embedding models (HuggingFace ids) — Phase 1, §3.2 of the implementation plan
EMBEDDING_MODELS = {
    "bge_m3": "BAAI/bge-m3",
    "indic_sbert": "l3cube-pune/indic-sentence-similarity-sbert",
    "multilingual_e5": "intfloat/multilingual-e5-large",
}

# Retrieval eval constants
TOP_K = 10          # Recall@k / nDCG@k cutoff
CMI_BINS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]   # for the diagnosis curve, §3.6
