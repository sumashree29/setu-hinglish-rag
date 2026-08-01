"""
Loads the 3 embedding arms used throughout the pipeline.
OWNER: R1 | PHASE: 1 (plan §3.2)
"""
from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODELS


def load_embedding_model(name: str) -> SentenceTransformer:
    """
    Args:
        name: one of "bge_m3", "indic_sbert", "multilingual_e5" (see config.py)

    Returns:
        a loaded SentenceTransformer. Cache these in the caller — don't reload
        per-query, each is 1-2GB.

    TODO (R1):
        model_id = EMBEDDING_MODELS[name]
        return SentenceTransformer(model_id)
        Note: first call downloads the model (~1-2GB) — this needs an internet
        connection to huggingface.co, so run this locally or in Colab, not
        inside a sandboxed environment with restricted network access.
    """
    raise NotImplementedError("R1: implement model loading + local caching")


def embed(texts, model: SentenceTransformer):
    """Thin wrapper around model.encode() — exists so every caller embeds
    the same way (e.g. same normalization/batch_size settings)."""
    raise NotImplementedError("R1: implement, decide normalize_embeddings=True/False once and document it")
