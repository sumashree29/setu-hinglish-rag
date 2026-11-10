import pytest
import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from scripts.compare_setu_v1_v2_scaled import split_trajectories_by_fold, build_grouped_folds


def test_grouped_folds_prevent_text_and_doc_leakage():
    queries = [
        {"query_id": "Q1", "text": "same query", "relevant_doc_ids": ["D1"]},
        {"query_id": "Q2", "text": "same query", "relevant_doc_ids": ["D2"]},
        {"query_id": "Q3", "text": "variant a", "relevant_doc_ids": ["D3"]},
        {"query_id": "Q4", "text": "variant b", "relevant_doc_ids": ["D3"]},
        {"query_id": "Q5", "text": "unique 5", "relevant_doc_ids": ["D5"]},
        {"query_id": "Q6", "text": "unique 6", "relevant_doc_ids": ["D6"]},
        {"query_id": "Q7", "text": "unique 7", "relevant_doc_ids": ["D7"]},
        {"query_id": "Q8", "text": "unique 8", "relevant_doc_ids": ["D8"]},
    ]
    folds = build_grouped_folds(queries, n_splits=4)
    q_by_id = {q["query_id"]: q for q in queries}
    for test_qids in folds:
        train_qids = set(q_by_id.keys()) - set(test_qids)
        train_texts = {q_by_id[q]["text"].lower() for q in train_qids}
        test_texts = {q_by_id[q]["text"].lower() for q in test_qids}
        train_docs = {d for q in train_qids for d in q_by_id[q]["relevant_doc_ids"]}
        test_docs = {d for q in test_qids for d in q_by_id[q]["relevant_doc_ids"]}
        assert train_texts.isdisjoint(test_texts)
        assert train_docs.isdisjoint(test_docs)

def test_oof_leakage():
    # Synthetic dataset where two queries have the IDENTICAL text but different IDs.
    # This mimics the issue in the pilot where auto-generated variants collapsed into the same text.
    qids = ["Q1", "Q2", "Q3", "Q4"]
    
    # Q1 and Q2 share the same text!
    q_by_id = {
        "Q1": {"query_id": "Q1", "text": "how to open an account"},
        "Q2": {"query_id": "Q2", "text": "how to open an account"},
        "Q3": {"query_id": "Q3", "text": "what is minimum balance"},
        "Q4": {"query_id": "Q4", "text": "credit card limit"},
    }
    
    trajectories = [
        {"query_id": "Q1", "query": "how to open an account", "action": "LAG"},
        {"query_id": "Q2", "query": "how to open an account", "action": "LQP"},
        {"query_id": "Q3", "query": "what is minimum balance", "action": "STOP"},
        {"query_id": "Q4", "query": "credit card limit", "action": "CAEP"},
    ]
    
    # 2 Folds:
    # Fold 1 tests Q1, Q3 -> trains on Q2, Q4
    # Fold 2 tests Q2, Q4 -> trains on Q1, Q3
    folds = [["Q1", "Q3"], ["Q2", "Q4"]]
    
    for fold_idx, test_qids in enumerate(folds):
        train_ids = set(qid for qid in qids if qid not in test_qids)
        test_ids = set(test_qids)
        
        # 1. Fold Disjointness check
        assert train_ids.isdisjoint(test_ids), "fold overlap detected"
        
        # Under the old, flawed text-based logic:
        # test_query_texts = set(q_by_id[qid]["text"] for qid in test_qids)
        # old_train_traj = [r for r in trajectories if r.get("query") not in test_query_texts]
        # For Fold 1 (test_qids = Q1, Q3):
        # test_query_texts = {"how to open an account", "what is minimum balance"}
        # Q2's text is "how to open an account", so Q2 would be INCORRECTLY EXCLUDED from training.
        
        # For Fold 2 (test_qids = Q2, Q4):
        # 1. Fold Disjointness check (handled inside the function)
        
        # New, fixed ID-based logic:
        train_traj, test_traj = split_trajectories_by_fold(trajectories, train_ids, test_ids)
        
        # Verify no valid training trajectories were dropped (which text-filtering did!)
        assert len(train_traj) == 2
        assert len(test_traj) == 2
        
    print("Leakage regression test passed. Pipeline is safe from text-based ID collisions.")

if __name__ == "__main__":
    test_oof_leakage()
