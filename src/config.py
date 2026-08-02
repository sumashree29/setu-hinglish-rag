"""
ACQR — central config.
Every module reads paths / constants from here so Phase 1-6 stay consistent.
"""
from pathlib import Path

# --- Paths ---
ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PILOT = ROOT / "data" / "pilot_corpus"
RESULTS_FIGS = ROOT / "results" / "figures"
RESULTS_LOGS = ROOT / "results" / "logs"

# --- Phase 1: Embedding models to compare ---
# NOTE: these need Hugging Face access, set up in a later task.
EMBEDDING_MODELS = {
    "bge_m3": "BAAI/bge-m3",
    "indic_sbert": "l3cube-pune/indic-sentence-similarity-sbert",  # matches Reference 3 verified link
    "me5_large": "intfloat/multilingual-e5-large",
}
# --- CMI / LID ---
# CMI (Code-Mixing Index) bands used throughout diagnosis + evaluation
CMI_BANDS = [
    (0, 10, "low"),
    (10, 25, "medium"),
    (25, 40, "high"),
    (40, 50, "very_high"),
]

# --- Retrieval ---
FAISS_INDEX_TYPE = "IndexFlatIP"  # cosine/IP similarity on normalized embeddings
TOP_K = 10

# --- Fusion (CARF) ---
CARF_ALPHA_DEFAULT = 0.5  # weight between dense score and CMI-adjusted rank; tuned in Phase 3

# --- SETU bandit controller ---
SETU_OPERATORS = ["LQP", "CAEP", "LAG"]  # order for v1 fixed baseline
BANDIT_EPSILON = 0.1

# --- Reproducibility ---
RANDOM_SEED = 42