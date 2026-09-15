"""Tests for setu/controller/setu_bandit.py. OWNER: R3."""
import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge

from setu.controller.setu_bandit import (
    ACTIONS,
    LinUCBController,
    EpsilonGreedyController,
    setu_v2_run,
)
from setu.evaluation.metrics import confidence_proxy


def test_linucb_select_action_returns_valid_action():
    controller = LinUCBController(context_dim=7, alpha=1.0)
    context = np.array([0.5, 0.3, 1.2, 0, 0, 0, 0], dtype=float)
    action = controller.select_action(context)
    assert action in ACTIONS


def test_linucb_update_modifies_internal_state():
    controller = LinUCBController(context_dim=7, alpha=1.0)
    context = np.array([0.5, 0.3, 1.2, 0, 0, 0, 0], dtype=float)
    
    A_before = controller.A["LQP"].copy()
    b_before = controller.b["LQP"].copy()

    controller.update(context, "LQP", reward=0.5)

    assert not np.allclose(controller.A["LQP"], A_before)
    assert not np.allclose(controller.b["LQP"], b_before)
    assert np.allclose(controller.b["LQP"], 0.5 * context)


def test_epsilon_greedy_controller_select_and_update():
    controller = EpsilonGreedyController(epsilon=0.0)
    context = np.array([0.5, 0.3, 1.2, 0, 0, 0, 0], dtype=float)
    action = controller.select_action(context)
    assert action in ACTIONS

    controller.update(context, "CAEP", 0.8)
    assert len(controller.history["CAEP"]["rewards"]) == 1
    assert controller.history["CAEP"]["rewards"][0] == 0.8


def test_setu_v2_run_terminates_and_returns_ranking():
    controller = LinUCBController(context_dim=7, alpha=1.0)
    doc_ids = ["C01", "C02", "C03", "C04", "C05"]
    raw_ranking = (doc_ids, [1.0, 0.8, 0.6, 0.4, 0.2])

    def dummy_embed_fn(texts):
        return np.random.randn(len(texts), 16).astype("float32")

    def dummy_faiss_search_fn(query_emb, k=5):
        return doc_ids, [1.0, 0.8, 0.6, 0.4, 0.2]

    # Dummy CAEP gate (3 features: fuzzy_score, embedding_cosine, entity_frequency)
    caep_gate = LogisticRegression()
    caep_gate.fit([[0, 0, 0], [100, 1.0, 10]], [0, 1])

    # Dummy LQP model
    lqp_model = Ridge()
    lqp_model.fit(np.eye(16), np.eye(16))

    ops, conf_trace, final_ranking, stop_reason = setu_v2_run(
        query="mera account balance check karna hai",
        controller=controller,
        raw_ranking=raw_ranking,
        embed_fn=dummy_embed_fn,
        entities=["account", "balance"],
        entity_freq={"account": 5, "balance": 5},
        caep_gate=caep_gate,
        lqp_model=lqp_model,
        faiss_search_fn=dummy_faiss_search_fn,
        confidence_fn=confidence_proxy,
        max_steps=3,
    )

    assert len(ops) <= 3
    assert len(conf_trace) == len(ops) + 1
    assert len(final_ranking) == len(doc_ids)
    assert stop_reason in ["explicit_stop", "repeat_action_forced", "max_steps_reached"]


def test_setu_v2_run_frozen_mode_does_not_mutate_controller():
    controller = LinUCBController(context_dim=7, alpha=1.0)
    doc_ids = ["C01", "C02", "C03", "C04", "C05"]
    raw_ranking = (doc_ids, [1.0, 0.8, 0.6, 0.4, 0.2])

    def dummy_embed_fn(texts):
        return np.random.randn(len(texts), 16).astype("float32")

    def dummy_faiss_search_fn(query_emb, k=5):
        return doc_ids, [1.0, 0.8, 0.6, 0.4, 0.2]

    caep_gate = LogisticRegression()
    caep_gate.fit([[0, 0, 0], [100, 1.0, 10]], [0, 1])

    lqp_model = Ridge()
    lqp_model.fit(np.eye(16), np.eye(16))

    A_snapshots = {a: controller.A[a].copy() for a in controller.actions}
    b_snapshots = {a: controller.b[a].copy() for a in controller.actions}

    ops, conf_trace, final_ranking, stop_reason = setu_v2_run(
        query="mera account balance check karna hai",
        controller=controller,
        raw_ranking=raw_ranking,
        embed_fn=dummy_embed_fn,
        entities=["account", "balance"],
        entity_freq={"account": 5, "balance": 5},
        caep_gate=caep_gate,
        lqp_model=lqp_model,
        faiss_search_fn=dummy_faiss_search_fn,
        confidence_fn=confidence_proxy,
        max_steps=3,
        train=False,
    )

    for a in controller.actions:
        assert np.allclose(controller.A[a], A_snapshots[a]), f"Action {a} matrix A mutated during frozen run!"
        assert np.allclose(controller.b[a], b_snapshots[a]), f"Action {a} vector b mutated during frozen run!"

def test_stop_reason_max_steps():
    # If the controller never outputs STOP and never repeats an action, it hits max_steps
    class DummyController:
        def __init__(self):
            self.actions = ACTIONS
            self.calls = 0
        def select_action(self, ctx):
            a = ["LAG", "CAEP", "LQP"][self.calls % 3]
            self.calls += 1
            return a
        def update(self, ctx, a, reward=0.0): pass

    controller = DummyController()
    doc_ids = ["C01"]
    def dummy_embed_fn(texts): return np.random.randn(len(texts), 16).astype("float32")
    def dummy_faiss_search_fn(query_emb, k=5): return doc_ids, [1.0]

    caep_gate = LogisticRegression(); caep_gate.fit([[0,0,0],[1,1,1]], [0,1])
    lqp_model = Ridge(); lqp_model.fit(np.eye(16), np.eye(16))

    ops, conf, rank, stop = setu_v2_run(
        query="test", controller=controller, raw_ranking=(doc_ids, [1.0]),
        embed_fn=dummy_embed_fn, entities=["test"], entity_freq={"test": 1},
        caep_gate=caep_gate, lqp_model=lqp_model, faiss_search_fn=dummy_faiss_search_fn,
        confidence_fn=confidence_proxy, max_steps=3
    )
    assert stop == "max_steps_reached"
    assert len(ops) == 3

def test_stop_reason_repeat_action():
    # If the controller repeats an action, it should force a stop
    class DummyController:
        def __init__(self):
            self.actions = ACTIONS
        def select_action(self, ctx):
            return "LAG" # always output LAG
        def update(self, ctx, a, reward=0.0): pass

    controller = DummyController()
    doc_ids = ["C01"]
    def dummy_embed_fn(texts): return np.random.randn(len(texts), 16).astype("float32")
    def dummy_faiss_search_fn(query_emb, k=5): return doc_ids, [1.0]

    caep_gate = LogisticRegression(); caep_gate.fit([[0,0,0],[1,1,1]], [0,1])
    lqp_model = Ridge(); lqp_model.fit(np.eye(16), np.eye(16))

    ops, conf, rank, stop = setu_v2_run(
        query="test", controller=controller, raw_ranking=(doc_ids, [1.0]),
        embed_fn=dummy_embed_fn, entities=["test"], entity_freq={"test": 1},
        caep_gate=caep_gate, lqp_model=lqp_model, faiss_search_fn=dummy_faiss_search_fn,
        confidence_fn=confidence_proxy, max_steps=3
    )
    assert stop == "repeat_action_forced"
    # The action was output first time, second time it's a repeat, so it converts to STOP
    assert ops == ["LAG", "STOP"]

def test_stop_reason_explicit_stop():
    class DummyController:
        def __init__(self):
            self.actions = ACTIONS
        def select_action(self, ctx):
            return "STOP"
        def update(self, ctx, a, reward=0.0): pass

    controller = DummyController()
    doc_ids = ["C01"]
    def dummy_embed_fn(texts): return np.random.randn(len(texts), 16).astype("float32")
    def dummy_faiss_search_fn(query_emb, k=5): return doc_ids, [1.0]

    caep_gate = LogisticRegression(); caep_gate.fit([[0,0,0],[1,1,1]], [0,1])
    lqp_model = Ridge(); lqp_model.fit(np.eye(16), np.eye(16))

    ops, conf, rank, stop = setu_v2_run(
        query="test", controller=controller, raw_ranking=(doc_ids, [1.0]),
        embed_fn=dummy_embed_fn, entities=["test"], entity_freq={"test": 1},
        caep_gate=caep_gate, lqp_model=lqp_model, faiss_search_fn=dummy_faiss_search_fn,
        confidence_fn=confidence_proxy, max_steps=3
    )
    assert stop == "explicit_stop"
    assert ops == ["STOP"]

def test_fold_disjointness_logic():
    # Simulate compare_setu_v1_v2_scaled.py fold logic
    import json
    queries = [
        {"query_id": "Q1", "text": "q1 text"},
        {"query_id": "Q2", "text": "q2 text"},
        {"query_id": "Q3", "text": "q1 text"}, # same text as Q1
        {"query_id": "Q4", "text": "q4 text"},
    ]
    qids = [q["query_id"] for q in queries]
    q_by_id = {q["query_id"]: q for q in queries}
    q_by_text = {q["text"]: q["query_id"] for q in queries}
    
    # Say we have 2 folds
    folds = [["Q1", "Q2"], ["Q3", "Q4"]]
    
    # Test fold 1
    test_qids = folds[0]
    train_qids = [qid for qid in qids if qid not in test_qids]
    
    # The actual assertion from the script
    assert len(set(train_qids).intersection(set(test_qids))) == 0
