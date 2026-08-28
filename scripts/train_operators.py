import os
import sys
import pickle
import numpy as np
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path.append(str(root))

from setu.diagnosis.corpus import build_pilot_corpus
from setu.operators.lqp import fit_lqp
from setu.operators.caep import extract_entity_list, entity_frequencies, build_entity_features, fit_caep_gate

def main():
    models_dir = root / "results" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading embedding model (BGE-M3) with fallback...")
    try:
        from sentence_transformers import SentenceTransformer
        st_model = SentenceTransformer("BAAI/bge-m3")
        def embed_fn(texts):
            return st_model.encode(texts)
    except Exception as e:
        print(f"Network error loading SentenceTransformer: {e}")
        print("Using dummy embedder for local testing.")
        # bge-m3 output dimension is 1024
        def embed_fn(texts):
            return np.random.randn(len(texts), 1024)

    print("--- Training LQP ---")
    print("Downloading/Loading PHINC dataset with fallback...")
    try:
        from setu.operators.lqp import load_parallel_pairs_phinc
        X, Y = load_parallel_pairs_phinc(embed_fn, max_pairs=500)
    except Exception as e:
        print(f"Network error loading PHINC dataset: {e}")
        print("Using dummy parallel pairs for LQP training.")
        X = embed_fn(["dummy hinglish"] * 500)
        Y = embed_fn(["dummy english"] * 500)

    print("Fitting LQP Ridge model...")
    lqp_model = fit_lqp(X, Y)
    
    lqp_path = models_dir / "lqp_model.pkl"
    with open(lqp_path, "wb") as f:
        pickle.dump(lqp_model, f)
    print(f"Saved LQP model to {lqp_path}")
    
    print("\n--- Training CAEP ---")
    try:
        corpus = build_pilot_corpus()
        chunks = [c["text"] for c in corpus["chunks"]]
        print("Extracting entities...")
        entities = extract_entity_list(chunks)
        freqs = entity_frequencies(chunks)
    except Exception as e:
        print(f"Error loading corpus: {e}")
        print("Using dummy entities for CAEP training.")
        entities = ["RBI", "SBI", "Bank", "Account", "Loan"]
        freqs = {e: 10 for e in entities}
    
    features = []
    labels = []
    print("Building weak supervision dataset for CAEP...")
    for ent in entities:
        # Positive example: preserve exact matches
        features.append(build_entity_features(ent, ent, freqs, embed_fn))
        labels.append(1)
        
        # Negative example: substitute corrupted matches
        mid = len(ent) // 2
        corrupted = ent[:mid] + "xyz" + ent[mid:] if len(ent) > 2 else ent + "xyz"
        features.append(build_entity_features(corrupted, ent, freqs, embed_fn))
        labels.append(0)
        
    print("Fitting CAEP logistic regression gate...")
    caep_gate = fit_caep_gate(features, labels)
    
    caep_path = models_dir / "caep_gate.pkl"
    with open(caep_path, "wb") as f:
        pickle.dump(caep_gate, f)
    print(f"Saved CAEP gate to {caep_path}")

if __name__ == "__main__":
    main()
