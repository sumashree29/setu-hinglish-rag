"""
SETU controller — the paper's central contribution.
OWNER: R3 | PHASE: 4 (plan §6)

SETU v1: fixed order, always all 4 steps (LAG -> CAEP -> LQP -> CARF), tau=1. Baseline.
SETU v2: learned contextual bandit that picks the operator sequence per query
         and halts adaptively based on a confidence signal.
"""
from typing import List, Dict, Tuple
import numpy as np

ACTIONS = ["LAG", "CAEP", "LQP", "STOP"]

"""
SETU controller — the paper's central contribution.
OWNER: R3 | PHASE: 4 (plan §6)

SETU v1: fixed order, always all 4 steps (LAG -> CAEP -> LQP -> CARF), tau=1. Baseline.
SETU v2: learned contextual bandit that picks the operator sequence per query
         and halts adaptively based on a confidence signal.
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
    raw_ranking: List[str],
    embed_fn,
    entities: List[str],
    entity_freq: Dict[str, int],
    caep_gate,
    lqp_model,
    faiss_search_fn,
) -> Dict:
    """
    Baseline: always LAG -> CAEP -> LQP -> CARF, no adaptivity.

    ASSUMPTION (flagged for team review): LAG has no query-rewriting executor
    built yet (only predict_strategy() exists, returning a label). Since v1 is
    explicitly the "run everything, no adaptivity" baseline, LAG's predicted
    strategy is logged as a diagnostic feature but does not gate/skip CAEP or
    LQP here -- CAEP (entity substitution) and LQP (embedding projection) are
    the two operators that actually transform the query in this baseline.

    Args:
        query: raw Hinglish query
        raw_ranking: FAISS top-k doc_ids for the *unmodified* query (baseline ranking)
        embed_fn: function(list[str]) -> embeddings, shared across operators
        entities, entity_freq: from setu.operators.caep.extract_entity_list/entity_frequencies
        caep_gate: fitted LogisticRegression from fit_caep_gate()
        lqp_model: fitted Ridge model from fit_lqp()
        faiss_search_fn: function(query_embedding) -> List[str] ranked doc_ids

    Returns:
        Dict with the trajectory (steps taken) and final fused ranking.
    """
    from setu.diagnosis.cmi import cmi
    from setu.diagnosis.lid_entropy import lid_entropy
    from setu.operators.lag import predict_strategy, entity_density, fit_lag_v1
    from setu.operators.caep import apply_caep
    from setu.fusion.carf import carf_v1

    cmi_score = cmi(query)
    entropy_score = lid_entropy(query)
    density = entity_density(query, entities)

    trajectory = []

    # Step 1: LAG -- diagnostic only (see assumption above)
    # NOTE: predict_strategy needs a fitted model; caller must have one ready.
    # Left as metadata-only here since no lag_model param was passed in yet --
    # this is a known gap to fill once R2 confirms LAG's intended pipeline role.
    trajectory.append({"action": "LAG", "note": "diagnostic only, no fitted model wired in yet"})

    # Step 2: CAEP -- entity-aware query rewrite
    corrected_query = apply_caep(query, entities, caep_gate, entity_freq, embed_fn=embed_fn)
    trajectory.append({"action": "CAEP", "output_query": corrected_query})

    # Step 3: LQP -- CMI-conditional embedding projection
    corrected_embedding = embed_fn([corrected_query])[0]
    projected_embedding = apply_lqp(np.asarray(corrected_embedding), cmi_score, lqp_model)
    corrected_ranking = faiss_search_fn(projected_embedding)
    trajectory.append({"action": "LQP", "corrected_ranking_top3": corrected_ranking[:3]})

    # Step 4: CARF -- fuse raw vs. corrected rankings
    fused_ranking = carf_v1(raw_ranking, corrected_ranking, cmi_score=cmi_score, cmi_max=1.0)
    trajectory.append({"action": "CARF", "fused_ranking_top3": fused_ranking[:3]})

    return {
        "query": query,
        "cmi": cmi_score,
        "lid_entropy": entropy_score,
        "entity_density": density,
        "trajectory": trajectory,
        "final_ranking": fused_ranking,
    }

class LinUCBController:
    """
    Contextual bandit: state = (CMI, LID-entropy, confidence, step t),
    actions = ACTIONS. Closed-form ridge-regression updates.
    TODO (R3): implement per plan §6.2. Simpler starting point (plan alt #3):
    epsilon-greedy value regression first, then upgrade to LinUCB if there's
    time. mabwiser is a fallback library if hand-implementing LinUCB is slow.
    """

    def __init__(self, n_actions: int = len(ACTIONS), context_dim: int = 4, alpha: float = 1.0):
        raise NotImplementedError("R3: init LinUCB matrices (A, b per action)")

    def select_action(self, context: np.ndarray) -> str:
        raise NotImplementedError("R3: implement UCB action selection")

    def update(self, context: np.ndarray, action: str, reward: float):
        raise NotImplementedError("R3: implement ridge-regression update")


def setu_v2_run(query: str, controller: LinUCBController) -> Tuple[List[str], List[float]]:
    """
    Run the learned controller end to end on one query: repeatedly select an
    action, apply the corresponding operator, update confidence, until STOP
    or a max-step cap is hit. Returns (operator sequence used, confidence trace).
    TODO (R3): implement after LinUCBController is trained.
    """
    raise NotImplementedError("R3: implement SETU v2 inference loop")
