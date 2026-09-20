import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import json
import numpy as np
from collections import Counter
from setu.diagnosis.cmi import cmi
from setu.diagnosis.lid_entropy import lid_entropy
from setu.operators.caep import extract_entity_list
from setu.operators.lag import fit_lag_v2, predict_strategy, entity_density

DATA_DIR = Path("data")

# 1. Load Data
corpus_v2_file = DATA_DIR / "processed" / "corpus_chunks_v2.jsonl"
queries_v3_file = DATA_DIR / "processed" / "queries_v3_final.json"
lag_labels_file = DATA_DIR / "processed" / "lag_labels_v3.json"

chunks_v2 = [json.loads(l) for l in open(corpus_v2_file, encoding='utf-8') if l.strip()]
queries_v3 = json.load(open(queries_v3_file, encoding='utf-8'))
lag_labeled_data = json.load(open(lag_labels_file, encoding='utf-8'))

doc_texts = [c["text"] for c in chunks_v2]
query_ids = [q["query_id"] for q in queries_v3]
q_by_id = {q["query_id"]: q for q in queries_v3}

query_cmi = {q["query_id"]: cmi(q["text"]) for q in queries_v3}
query_entropy = {q["query_id"]: lid_entropy(q["text"]) for q in queries_v3}
entities = extract_entity_list(doc_texts)

n_splits = 5
folds = np.array_split(query_ids, n_splits)

all_predictions = []

for fold_idx, test_qids in enumerate(folds):
    train_ids = set(query_ids) - set(test_qids)
    test_ids = set(test_qids)
    
    fold_lag_data = [d for d in lag_labeled_data if d["query_id"] in train_ids]
    
    X_lag = np.array([[d["cmi"], d["lid_entropy"], d["entity_density"]] for d in fold_lag_data])
    y_lag = np.array([d["label"] for d in fold_lag_data])
    
    fold_lag_model = fit_lag_v2(X_lag, y_lag)
    
    for qid in test_qids:
        q_text = q_by_id[qid]["text"]
        q_c = query_cmi[qid]
        q_ent = query_entropy[qid]
        q_dens = entity_density(q_text, entities)
        
        lag_strat = predict_strategy(q_c, q_ent, q_dens, fold_lag_model)
        all_predictions.append(lag_strat)

dist = Counter(all_predictions)
print(f"Strategy Distribution across {len(all_predictions)} queries:")
for k, v in dist.items():
    print(f"{k}: {v} ({v/len(all_predictions)*100:.1f}%)")
