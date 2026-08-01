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


def log_trajectory(query: str, state: Dict, action: str, confidence_before: float, confidence_after: float) -> Dict:
    """
    One row of the trajectory log = this controller's training data.
    TODO (R3): append rows to data/logs/trajectories.jsonl (plain append-only
    log is enough at this scale per plan §6.1 alt #2). State should include
    (CMI, LID-entropy, confidence, step t).
    """
    raise NotImplementedError("R3: implement trajectory logging")


def setu_v1_fixed_order(query: str, state: Dict) -> List[str]:
    """Baseline: always LAG -> CAEP -> LQP -> CARF, no adaptivity."""
    raise NotImplementedError("R3: implement fixed baseline pipeline")


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
