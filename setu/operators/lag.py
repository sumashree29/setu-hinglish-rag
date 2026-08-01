"""
LAG — Learned Adaptive Gating.
OWNER: R2 | PHASE: 2 (plan §4.3)

Predicts which correction sub-strategy (light-normalize / dual-variant /
full-rewrite) to use for a given query, based on [CMI, LID-entropy, entity-density].
"""
from typing import List
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression


SUB_STRATEGIES = ["light_normalize", "dual_variant", "full_rewrite"]


def label_pilot_queries(pilot_queries: List[str], phase1_results) -> List[int]:
    """
    For 300-500 pilot queries, label which sub-strategy actually helped
    retrieval in Phase 1's pilot runs (this requires Phase 1 results to exist first).
    TODO (R2): if labeling 300-500 is too slow, start with 150 and bootstrap
    with the classifier's own high-confidence pseudo-labels (plan alt #3),
    manually spot-checking pseudo-labels before trusting them.
    """
    raise NotImplementedError("R2: implement labeling workflow, run after Phase 1")


def fit_lag_v1(features, labels) -> LogisticRegression:
    """Simple baseline: sklearn LogisticRegression."""
    raise NotImplementedError("R2: fit LAG v1")


def fit_lag_v2(features, labels) -> lgb.LGBMClassifier:
    """Reported model: LightGBM classifier + feature importance plot.
    Alt: xgboost as drop-in if LightGBM install issues (plan alt #1)."""
    raise NotImplementedError("R2: fit LAG v2, report accuracy/F1/feature importance")


def predict_strategy(cmi: float, lid_entropy: float, entity_density: float, model) -> str:
    """Returns one of SUB_STRATEGIES."""
    raise NotImplementedError("R2: implement prediction")
