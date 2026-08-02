"""
SETU — central config.
Every module reads paths / constants from here so Phase 1-6 stay consistent.
"""
from pathlib import Path

# --- Paths (matches the actual folder structure already used by Phase 1 scripts) ---
ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PILOT = ROOT / "data" / "pilot_corpus"
RESULTS_FIGS = ROOT / "results" / "figures"
RESULTS_LOGS = ROOT / "results" / "logs"

# --- Phase 1: Embedding models to compare ---
EMBEDDING_MODELS = {
    "bge_m3": "BAAI/bge-m3",
    "indic_sbert": "l3cube-pune/indic-sentence-similarity-sbert",
    "me5_large": "intfloat/multilingual-e5-large",
}

# --- CMI / LID ---
# CMI is a balance measure, maxes at 50 for two-language mixing (NOT a 0-1 scale).
CMI_BANDS = [
    (0, 10, "low"),
    (10, 25, "medium"),
    (25, 40, "high"),
    (40, 50, "very_high"),
]

# --- Retrieval ---
FAISS_INDEX_TYPE = "IndexFlatIP"
TOP_K = 10

# --- Fusion (CARF) ---
CARF_ALPHA_DEFAULT = 0.5

# --- SETU bandit controller ---
SETU_OPERATORS = ["LQP", "CAEP", "LAG"]
BANDIT_EPSILON = 0.1

# --- Reproducibility ---
RANDOM_SEED = 42