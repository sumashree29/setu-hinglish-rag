"""
SETU controller — the paper's central contribution.
OWNER: R3 | PHASE: 4 (plan §6)

SETU v1: fixed order, always all 4 steps (LAG -> CAEP -> LQP -> CARF), tau=1. Baseline.
SETU v2: learned contextual bandit that picks the operator sequence per query
         and halts adaptively based on a confidence signal.

CONVENTION: any `faiss_search_fn` passed into functions below is expected to
return a (ranked_doc_ids, scores) tuple -- List[str], List[float] -- matching
the shape our Phase 1 run_retrieval.py already stores. This matters because
confidence_proxy() (setu/evaluation/metrics.py) needs raw similarity scores,
not doc IDs.
"""
import json
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np

ACTIONS = ["LAG", "CAEP", "LQP", "STOP"]

TRAJECTORY_LOG_PATH = Path("data/logs/trajectories.jsonl")


def log_trajectory(query: str, state: Dict, action: str, confidence_before: float, confidence_after: float) -> Dict:
    """
    One row of the trajectory log = this controller's training data.
    Appends to data/logs/trajectories.jsonl (append-only, per plan §6.1 alt #2).
    State should include (CMI, LID-entropy, confidence, step t).
    """
    TRAJECTORY_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    row = {
        "query": query,
        "state": state,
        "action": action,
        "confidence_before": confidence_before,
        "confidence_after": confidence_after,
        "reward": confidence_after - confidence_before,
    }

    with open(TRAJECTORY_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")

    return row


def setu_v1_fixed_order(
    query: str,
    raw_ranking: Tuple[List[str], List[float]],
    embed_fn,
    entities: List[str],
    entity_freq: Dict[str, int],
    caep_gate,
    lqp_model,
    faiss_search_fn,
    lag_model=None,
) -> Dict:
    """
    Baseline: always LAG -> CAEP -> LQP -> CARF, no adaptivity.

    Args:
        query: raw Hinglish query
        raw_ranking: (ranked_doc_ids, scores) for the *unmodified* query
        embed_fn: function(list[str]) -> embeddings, shared across operators
        entities, entity_freq: from setu.operators.caep.extract_entity_list/entity_frequencies
        caep_gate: fitted LogisticRegression from fit_caep_gate()
        lqp_model: fitted Ridge model from fit_lqp()
        faiss_search_fn: function(query_embedding) -> (ranked_doc_ids, scores)
        lag_model: optional fitted classifier for LAG predict_strategy()

    Returns:
        Dict with the trajectory (steps taken) and final fused ranking.
    """
    from setu.diagnosis.cmi import cmi
    from setu.diagnosis.lid_entropy import lid_entropy
    from setu.operators.lag import entity_density, predict_strategy, apply_lag
    from setu.operators.caep import apply_caep
    from setu.operators.lqp import apply_lqp
    from setu.fusion.carf import carf_v1, rrf_baseline

    cmi_score = cmi(query)
    entropy_score = lid_entropy(query)
    density = entity_density(query, entities)

    raw_doc_ids, raw_scores = raw_ranking

    trajectory = []
    current_query = query

    # Step 1: LAG -- learned adaptive gating
    if lag_model is not None:
        strategy = predict_strategy(cmi_score, entropy_score, density, lag_model)
        lag_out = apply_lag(current_query, strategy, entities=entities, embed_fn=embed_fn, caep_gate=caep_gate, entity_freq=entity_freq)
        if isinstance(lag_out, list):
            q1_emb = embed_fn([lag_out[0]])[0]
            q2_emb = embed_fn([lag_out[1]])[0]
            r1_ids, r1_scores = faiss_search_fn(np.asarray(q1_emb))
            r2_ids, r2_scores = faiss_search_fn(np.asarray(q2_emb))
            merged_ranking = rrf_baseline([r1_ids, r2_ids])
            trajectory.append({"action": "LAG", "strategy": strategy, "dual_queries": lag_out, "merged_ranking_top3": merged_ranking[:3]})
            current_query = lag_out[1]
        else:
            current_query = lag_out
            lag_emb = embed_fn([current_query])[0]
            lag_doc_ids, lag_scores = faiss_search_fn(np.asarray(lag_emb))
            trajectory.append({"action": "LAG", "strategy": strategy, "output_query": current_query, "ranking_top3": lag_doc_ids[:3]})
    else:
        trajectory.append({"action": "LAG", "note": "diagnostic only, no fitted model wired in yet"})

    # Step 2: CAEP -- entity-aware query rewrite
    corrected_query = apply_caep(current_query, entities, caep_gate, entity_freq, embed_fn=embed_fn)
    trajectory.append({"action": "CAEP", "output_query": corrected_query})

    # Step 3: LQP -- CMI-conditional embedding projection
    corrected_embedding = embed_fn([corrected_query])[0]
    projected_embedding = apply_lqp(np.asarray(corrected_embedding), cmi_score, lqp_model)
    corrected_doc_ids, corrected_scores = faiss_search_fn(projected_embedding)
    trajectory.append({"action": "LQP", "corrected_ranking_top3": corrected_doc_ids[:3]})

    # Step 4: CARF -- fuse raw vs. corrected rankings
    fused_ranking = carf_v1(raw_doc_ids, corrected_doc_ids, cmi_score=cmi_score, cmi_max=1.0)
    trajectory.append({"action": "CARF", "fused_ranking_top3": fused_ranking[:3]})

    return {
        "query": query,
        "cmi": cmi_score,
        "lid_entropy": entropy_score,
        "entity_density": density,
        "trajectory": trajectory,
        "final_ranking": fused_ranking,
    }


class EpsilonGreedyController:
    """
    Simple first working policy for SETU v2 (plan §6.2 alt #3): epsilon-greedy
    value regression. Validates the full v2 loop end-to-end before adding
    LinUCB's ridge-regression complexity on top.

    Same interface as LinUCBController below (select_action, update) so
    setu_v2_run() can accept either without changes -- this is meant to be
    upgraded to LinUCB later, not replaced by it.
    """

    def __init__(self, actions: List[str] = None, epsilon: float = 0.1):
        self.actions = actions if actions is not None else ACTIONS
        self.epsilon = epsilon
        self.history = {a: {"contexts": [], "rewards": []} for a in self.actions}

    def select_action(self, context: np.ndarray) -> str:
        """
        Epsilon-greedy with linear regression on historical (context, reward) pairs.
        Falls back to uniform random exploration when epsilon triggers OR when
        an action hasn't been tried enough times to fit a regression line.
        """
        import random

        if random.random() < self.epsilon:
            return random.choice(self.actions)

        # Predict reward for each action using a simple Ridge model fit on history
        best_action = None
        best_pred = -float("inf")

        for action in self.actions:
            h = self.history[action]
            if len(h["rewards"]) < 3:
                # Not enough data to fit regression -- treat as unknown/high-value to explore
                pred = random.random()
            else:
                try:
                    from sklearn.linear_model import Ridge
                    clf = Ridge(alpha=1.0)
                    clf.fit(np.asarray(h["contexts"]), np.asarray(h["rewards"]))
                    pred = float(clf.predict(context.reshape(1, -1))[0])
                except Exception:
                    pred = float(np.mean(h["rewards"]))

            if pred > best_pred:
                best_pred = pred
                best_action = action

        return best_action or random.choice(self.actions)

    def update(self, context: np.ndarray, action: str, reward: float):
        if action in self.history:
            self.history[action]["contexts"].append(context.tolist())
            self.history[action]["rewards"].append(reward)


class LinUCBController:
    """
    Plan §6.2 LinUCB controller stub.
    Keep the interface identical to EpsilonGreedyController so callers can swap.
    Filled as a stub so Phase 4 tests can import it if needed, but the actual
    hand-implementing LinUCB is slow.
    """

    def __init__(self, n_actions: int = len(ACTIONS), context_dim: int = 4, alpha: float = 1.0):
        raise NotImplementedError("R3: init LinUCB matrices (A, b per action)")

    def select_action(self, context: np.ndarray) -> str:
        raise NotImplementedError("R3: implement UCB action selection")

    def update(self, context: np.ndarray, action: str, reward: float):
        raise NotImplementedError("R3: implement ridge-regression update")


def setu_v2_run(
    query: str,
    controller,
    raw_ranking: Tuple[List[str], List[float]],
    embed_fn,
    entities: List[str],
    entity_freq: Dict[str, int],
    caep_gate,
    lqp_model,
    faiss_search_fn,
    confidence_fn,
    max_steps: int = 4,
    lag_model=None,
) -> Tuple[List[str], List[float], List[str]]:
    """
    Run the learned controller end to end on one query: repeatedly select an
    action, apply the corresponding operator, update confidence, until STOP
    or a max-step cap is hit. Returns (operator sequence used, confidence trace).

    Args:
        raw_ranking: (ranked_doc_ids, scores) for the *unmodified* query
        faiss_search_fn: function(query_embedding) -> (ranked_doc_ids, scores)
        confidence_fn: setu.evaluation.metrics.confidence_proxy or compatible
            -- takes a List[float] of scores (NOT doc IDs) and a method name,
            returns a single float confidence value.
        lag_model: optional fitted classifier for LAG predict_strategy()
    """
    from setu.diagnosis.cmi import cmi
    from setu.diagnosis.lid_entropy import lid_entropy
    from setu.operators.lag import entity_density
    from setu.operators.caep import apply_caep
    from setu.operators.lqp import apply_lqp
    from setu.fusion.carf import carf_v1

    cmi_score = cmi(query)
    entropy_score = lid_entropy(query)
    density = entity_density(query, entities)

    current_query = query
    current_ranking, current_scores = raw_ranking
    confidence = confidence_fn(current_scores, method="margin")

    operator_sequence = []
    confidence_trace = [confidence]

    for step in range(max_steps):
        context = np.array([cmi_score, entropy_score, confidence, step], dtype=float)
        action = controller.select_action(context)
        if action == "STOP":
            controller.update(context, "STOP", reward=0.0)
            operator_sequence.append("STOP")
            break
       

        confidence_before = confidence

        if action == "CAEP":
            current_query = apply_caep(current_query, entities, caep_gate, entity_freq, embed_fn=embed_fn)
            current_embedding = embed_fn([current_query])[0]
            current_ranking, current_scores = faiss_search_fn(np.asarray(current_embedding))

        elif action == "LQP":
            current_embedding = embed_fn([current_query])[0]
            projected_embedding = apply_lqp(np.asarray(current_embedding), cmi_score, lqp_model)
            current_ranking, current_scores = faiss_search_fn(projected_embedding)

        elif action == "LAG":
            if lag_model is not None:
                from setu.operators.lag import predict_strategy, apply_lag
                from setu.fusion.carf import rrf_baseline
                strategy = predict_strategy(cmi_score, entropy_score, density, lag_model)
                lag_out = apply_lag(current_query, strategy, entities=entities, embed_fn=embed_fn, caep_gate=caep_gate, entity_freq=entity_freq)
                if isinstance(lag_out, list):
                    q1_emb = embed_fn([lag_out[0]])[0]
                    q2_emb = embed_fn([lag_out[1]])[0]
                    r1_ids, r1_scores = faiss_search_fn(np.asarray(q1_emb))
                    r2_ids, r2_scores = faiss_search_fn(np.asarray(q2_emb))
                    current_ranking = rrf_baseline([r1_ids, r2_ids])
                    current_scores = [1.0 / (idx + 1) for idx in range(len(current_ranking))]
                    current_query = lag_out[1]
                else:
                    current_query = lag_out
                    current_embedding = embed_fn([current_query])[0]
                    current_ranking, current_scores = faiss_search_fn(np.asarray(current_embedding))
            else:
                # Diagnostic only -- see setu_v1_fixed_order's note on LAG's gap.
                # No query transformation happens here; ranking/scores unchanged.
                pass

        confidence = confidence_fn(current_scores, method="margin")
        reward = confidence - confidence_before

        controller.update(context, action, reward)
        log_trajectory(
            query=query,
            state={"cmi": cmi_score, "lid_entropy": entropy_score, "confidence": confidence_before, "step": step},
            action=action,
            confidence_before=confidence_before,
            confidence_after=confidence,
        )

        operator_sequence.append(action)
        confidence_trace.append(confidence)

    fused_ranking = carf_v1(raw_ranking[0], current_ranking, cmi_score=cmi_score, cmi_max=1.0)

    return operator_sequence, confidence_trace, fused_ranking