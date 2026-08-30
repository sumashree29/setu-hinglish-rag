import os
import sys
import json
import pickle
import numpy as np
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path.append(str(root))

from setu.diagnosis.corpus import build_pilot_corpus
from setu.operators.lqp import fit_lqp
from setu.operators.caep import extract_entity_list, entity_frequencies, build_entity_features, fit_caep_gate
from rapidfuzz import fuzz

def main():
    models_dir = root / "results" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading embedding model (BGE-M3)...")
    try:
        from sentence_transformers import SentenceTransformer
        st_model = SentenceTransformer("BAAI/bge-m3")
        def embed_fn(texts):
            return st_model.encode(texts)
    except Exception as e:
        raise RuntimeError(f"Failed to load real training data/embeddings: {e}. Refusing to train on dummy data.")

    print("--- Training LQP ---")
    print("Downloading/Loading PHINC dataset...")
    try:
        from setu.operators.lqp import load_parallel_pairs_phinc
        X, Y = load_parallel_pairs_phinc(embed_fn, max_pairs=500)
    except Exception as e:
        raise RuntimeError(f"Failed to load real training data/embeddings: {e}. Refusing to train on dummy data.")

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
        raise RuntimeError(f"Failed to load real training data/embeddings: {e}. Refusing to train on dummy data.")
    
    features = []
    labels = []
    print("Building weak supervision dataset for CAEP (label 1=substitute, label 0=preserve/reject)...")
    common_hard_negatives = [
        "karna", "kaise", "hota", "milta", "paise", "kistein", "family", "annual",
        "benefit", "mandatory", "online", "yojana", "status", "hai", "kya", "mein",
        "wala", "wali", "purana", "naye", "scheme", "document", "apply", "portal",
        "open", "close", "minimum", "limit", "amount", "yearly", "charge", "valid",
    ]
    for ent in entities:
        # 1. Exact match -> Label 0 (Preserve, no substitution needed)
        features.append(build_entity_features(ent, ent, freqs, embed_fn))
        labels.append(0)

        # 2. Positive substitution candidates -> Label 1 (Substitute with canonical entity)
        # Lowercase / case variants
        if ent.lower() != ent:
            features.append(build_entity_features(ent.lower(), ent, freqs, embed_fn))
            labels.append(1)

        # Phonetic / typo variants (deletion, duplication, vowel swap)
        if len(ent) > 3:
            del_var = ent[:2] + ent[3:]
            features.append(build_entity_features(del_var.lower(), ent, freqs, embed_fn))
            labels.append(1)

        vowel_var = ent.lower().replace("aa", "a").replace("ee", "i").replace("oo", "u")
        if vowel_var != ent.lower():
            features.append(build_entity_features(vowel_var, ent, freqs, embed_fn))
            labels.append(1)

        dup_var = ent.lower() + ent[-1].lower()
        features.append(build_entity_features(dup_var, ent, freqs, embed_fn))
        labels.append(1)

        # Domain-specific known variants
        if "Aadhaar" in ent:
            for v in ["aadhar", "adhar", "aaddhar"]:
                features.append(build_entity_features(v, ent, freqs, embed_fn))
                labels.append(1)
        if "BSBDA" in ent:
            for v in ["bsb-da", "bsbdaa", "bsbda"]:
                features.append(build_entity_features(v, ent, freqs, embed_fn))
                labels.append(1)
        if "PM-KISAN" in ent or "Kisan" in ent:
            for v in ["pmkisan", "pm-kisaan", "kisaan"]:
                features.append(build_entity_features(v, ent, freqs, embed_fn))
                labels.append(1)
        if "KYC" in ent:
            features.append(build_entity_features("kyc", ent, freqs, embed_fn))
            labels.append(1)

        # 3. Hard negative words -> Label 0 (Do not substitute unrelated words)
        for hw in common_hard_negatives:
            score = fuzz.ratio(hw.lower(), ent.lower())
            if score < 70:
                features.append(build_entity_features(hw, ent, freqs, embed_fn))
                labels.append(0)

    print(f"Fitting CAEP logistic regression gate on {len(features)} examples (Pos: {sum(labels)}, Neg: {len(labels)-sum(labels)})...")
    caep_gate = fit_caep_gate(features, labels)
    
    caep_path = models_dir / "caep_gate.pkl"
    with open(caep_path, "wb") as f:
        pickle.dump(caep_gate, f)
    print(f"Saved CAEP gate to {caep_path}")

    print("\n--- Training LAG ---")
    lag_labels_path = root / "data" / "processed" / "lag_labels.json"
    if not lag_labels_path.exists():
        print("Generating lag_labels.json...")
        from scripts.label_lag_queries import main as generate_lag_labels
        generate_lag_labels()

    with open(lag_labels_path, "r", encoding="utf-8") as f:
        labels_data = json.load(f)

    X_lag = np.array([[d["cmi"], d["lid_entropy"], d["entity_density"]] for d in labels_data])
    y_lag = np.array([d["label"] for d in labels_data])

    print(f"Fitting LAG LightGBM classifier on {len(labels_data)} labeled queries...")
    from setu.operators.lag import fit_lag_v2
    lag_model = fit_lag_v2(X_lag, y_lag)

    lag_path = models_dir / "lag_model.pkl"
    with open(lag_path, "wb") as f:
        pickle.dump(lag_model, f)
    print(f"Saved LAG model to {lag_path}")

    print("\n--- Training LinUCB Policy ---")
    traj_path = root / "data" / "logs" / "trajectories.jsonl"
    if traj_path.exists():
        from setu.controller.setu_bandit import LinUCBController
        print(f"Pre-training LinUCB policy on offline trajectories ({traj_path})...")
        linucb_controller = LinUCBController(context_dim=7, alpha=1.0)
        linucb_controller.fit_from_trajectories(traj_path)
        policy_path = models_dir / "linucb_policy.pkl"
        linucb_controller.save(policy_path)
        print(f"Saved LinUCB policy to {policy_path}")
    else:
        print("No trajectory log found. Skipping offline LinUCB policy fit.")

if __name__ == "__main__":
    main()
