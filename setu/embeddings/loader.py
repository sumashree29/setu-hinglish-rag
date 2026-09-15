"""
Loads the 3 embedding arms used throughout the pipeline.
OWNER: R1 | PHASE: 1 (plan §3.2)
"""
import os
from sentence_transformers import SentenceTransformer
from setu.config import EMBEDDING_MODELS, MODEL_REVISIONS


def load_embedding_model(name: str) -> SentenceTransformer:
    """
    Args:
        name: one of "bge_m3", "indic_sbert", "me5_large", "mcontriever"
    Returns:
        a loaded SentenceTransformer with pinned revision.
    """
    if name not in EMBEDDING_MODELS:
        raise ValueError(f"Unknown model name: {name}")
    
    model_id = EMBEDDING_MODELS[name]
    revision = MODEL_REVISIONS.get(model_id, None)
    
    local_files_only = os.environ.get("SETU_LOCAL_FILES_ONLY", "False").lower() in ("true", "1", "yes")
    
    return SentenceTransformer(
        model_id, 
        revision=revision,
        local_files_only=local_files_only
    )


def embed(texts, model: SentenceTransformer):
    """Thin wrapper around model.encode() — exists so every caller embeds
    the same way (e.g. same normalization/batch_size settings)."""
    return model.encode(
        texts, 
        batch_size=32, 
        normalize_embeddings=True, 
        convert_to_numpy=True
    )
