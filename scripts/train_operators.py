import os
import sys
import pickle
from pathlib import Path
from sentence_transformers import SentenceTransformer

root = Path(__file__).resolve().parents[1]
sys.path.append(str(root))

from setu.diagnosis.corpus import build_pilot_corpus
from setu.operators.lqp import load_parallel_pairs_phinc, fit_lqp
from setu.operators.caep import extract_entity_list, entity_frequencies, build_entity_features, fit_caep_gate

def main():
    root = Path(__file__).resolve().parents[1]
    models_dir = root / "results" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading embedding model (BGE-M3)...")
    st_model = SentenceTransformer("BAAI/bge-m3")
    
    def embed_fn(texts):
        return st_model.encode(texts)
        
    print("--- Training LQP ---")
    print("Downloading/Loading PHINC dataset (max 500 pairs for speed)...")
    X, Y = load_parallel_pairs_phinc(embed_fn, max_pairs=500)
    print("Fitting LQP Ridge model...")
    lqp_model = fit_lqp(X, Y)
    
    lqp_path = models_dir / "lqp_model.pkl"
    with open(lqp_path, "wb") as f:
        pickle.dump(lqp_model, f)
    print(f"Saved LQP model to {lqp_path}")
    
    print("\n--- Training CAEP ---")
    corpus = build_pilot_corpus()
    chunks = [c["text"] for c in corpus["chunks"]]
    print("Extracting entities...")
    entities = extract_entity_list(chunks)
    freqs = entity_frequencies(chunks)
    
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
