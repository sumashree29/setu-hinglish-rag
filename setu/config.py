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
# CMI (Code-Mixing Index) bands, calibrated for the cmi() function's [0,1]
# range (fraction of non-majority-language tokens, per plan §3.5).
CMI_BANDS = [
    (0, 0.15, "low"),
    (0.15, 0.35, "medium"),
    (0.35, 0.55, "high"),
    (0.55, 1.01, "very_high"),
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