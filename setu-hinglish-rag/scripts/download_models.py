"""
Run this once (locally or in Colab — needs real internet access to
huggingface.co, which a sandboxed dev environment may not have).
OWNER: R1 | PHASE: 1
"""
from setu.embeddings.loader import load_embedding_model
from config import EMBEDDING_MODELS

if __name__ == "__main__":
    for name in EMBEDDING_MODELS:
        print(f"Downloading {name}...")
        load_embedding_model(name)
    print("All embedding models cached locally.")
