"""
Train and persist per-backbone operator checkpoints into results/models/:
  - lqp_model_{model_key}.pkl (Ridge regression on 500 PHINC parallel pairs)
  - caep_gate_{model_key}.pkl (LogisticRegression entity gate on corpus entities + synthetic variants)
  - lag_model_v3.pkl (LightGBM strategy classifier on data/processed/lag_labels_v3.json)
"""
import gc
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from rapidfuzz import fuzz

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from setu.config import set_seed
from setu.embeddings.loader import load_embedding_model, embed
from setu.operators.lqp import fit_lqp
from setu.operators.caep import extract_entity_list, entity_frequencies, build_entity_features, fit_caep_gate
from setu.operators.lag import fit_lag_v2

COMMON_HARD_NEGATIVES = [
    "karna", "kaise", "hota", "milta", "paise", "kistein", "family", "annual",
    "benefit", "mandatory", "online", "yojana", "status", "hai", "kya", "mein",
    "wala", "wali", "purana", "naye", "scheme", "document", "apply", "portal",
    "open", "close", "minimum", "limit", "amount", "yearly", "charge", "valid",
]


def load_phinc_pairs(max_pairs: int = 500):
    from datasets import load_dataset
    ds = load_dataset("LingoIITGN/PHINC", split="train")
    ds = ds.select(range(min(max_pairs, len(ds))))
    h_col = "Sentence" if "Sentence" in ds.column_names else "Hinglish Code-Mixed Sentence"
    e_col = "English_Translation" if "English_Translation" in ds.column_names else "Human Translated English Sentence"
    return [row[h_col] for row in ds], [row[e_col] for row in ds]


def build_caep_training_data(entities, freqs, embed_fn):
    features, labels = [], []
    for ent in entities:
        features.append(build_entity_features(ent, ent, freqs, embed_fn))
        labels.append(0)

        if ent.lower() != ent:
            features.append(build_entity_features(ent.lower(), ent, freqs, embed_fn))
            labels.append(1)

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

        for hw in COMMON_HARD_NEGATIVES:
            if fuzz.ratio(hw.lower(), ent.lower()) < 70:
                features.append(build_entity_features(hw, ent, freqs, embed_fn))
                labels.append(0)
    return features, labels


def main():
    set_seed()
    models_dir = ROOT / "results" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    chunks = [json.loads(l) for l in open(ROOT / "data/processed/corpus_chunks_v2.jsonl", encoding="utf-8") if l.strip()]
    doc_texts = [c["text"] for c in chunks]
    entities = extract_entity_list(doc_texts)
    freqs = entity_frequencies(doc_texts)
    print(f"Extracted {len(entities)} entities from {len(doc_texts)} chunks.")

    # Train global LAG reference checkpoint (note: evaluation scripts fit per-fold OOF LAG)
    lag_labels = json.load(open(ROOT / "data/processed/lag_labels_v3.json", encoding="utf-8"))
    X_lag = np.array([[d["cmi"], d["lid_entropy"], d["entity_density"]] for d in lag_labels])
    y_lag = np.array([d["label"] for d in lag_labels])
    lag_model = fit_lag_v2(X_lag, y_lag)
    with open(models_dir / "lag_model_v3.pkl", "wb") as f:
        pickle.dump(lag_model, f)
    print("Saved lag_model_v3.pkl")

    hinglish_sents, english_sents = load_phinc_pairs(max_pairs=500)
    print(f"Loaded {len(hinglish_sents)} PHINC parallel pairs for LQP.")

    backbones = {
        "bge_m3": False,
        "indic_sbert": False,
        "me5_large": True,
    }

    for model_key, needs_prefix in backbones.items():
        lqp_path = models_dir / f"lqp_model_{model_key}.pkl"
        caep_path = models_dir / f"caep_gate_{model_key}.pkl"
        if lqp_path.exists() and caep_path.exists():
            print(f"[{model_key}] Checkpoints already exist; skipping.")
            continue

        print(f"\n[{model_key}] Loading backbone...")
        st_model = load_embedding_model(model_key)

        embed_cache = {}

        def embed_fn(texts):
            if len(texts) <= 4 and all(t in embed_cache for t in texts):
                return np.stack([embed_cache[t] for t in texts])
            inp = [f"query: {t}" for t in texts] if needs_prefix else texts
            vecs = embed(inp, st_model).astype("float32")
            for t, v in zip(texts, vecs):
                if len(embed_cache) < 5000:
                    embed_cache[t] = v
            return vecs

        if not lqp_path.exists():
            X = embed_fn(hinglish_sents)
            Y = embed_fn(english_sents)
            lqp_model = fit_lqp(X, Y, alpha_reg=1.0)
            with open(lqp_path, "wb") as f:
                pickle.dump(lqp_model, f)
            print(f"[{model_key}] Saved {lqp_path.name}")

        if not caep_path.exists():
            # Use pilot + top frequent entities for compact, fast training
            pilot_chunks = [c["text"] for c in chunks[:20]]
            pilot_entities = extract_entity_list(pilot_chunks)
            top_entities = sorted(entities, key=lambda e: freqs.get(e, 0), reverse=True)[:40]
            train_entities = sorted(set(pilot_entities) | set(top_entities))
            feats, labs = build_caep_training_data(train_entities, freqs, embed_fn)
            caep_gate = fit_caep_gate(feats, labs)
            with open(caep_path, "wb") as f:
                pickle.dump(caep_gate, f)
            print(f"[{model_key}] Saved {caep_path.name} ({len(feats)} examples)")

        del st_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\nAll operator checkpoints built successfully.")


if __name__ == "__main__":
    main()
