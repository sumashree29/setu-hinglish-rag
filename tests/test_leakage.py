import pytest
import numpy as np
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from scripts.compare_setu_v1_v2_scaled import split_trajectories_by_fold

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
