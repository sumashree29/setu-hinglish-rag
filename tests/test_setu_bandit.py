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

    ops, conf_trace, final_ranking = setu_v2_run(
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
