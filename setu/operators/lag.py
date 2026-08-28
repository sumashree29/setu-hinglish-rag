"""
LAG — Learned Adaptive Gating.
OWNER: R2 | PHASE: 2 (plan §4.3)

Predicts which correction sub-strategy (light-normalize / dual-variant /
full-rewrite) to use for a given query, based on [CMI, LID-entropy, entity-density].
"""
from typing import List, Dict, Union
import numpy as np
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression

SUB_STRATEGIES = ["light_normalize", "dual_variant", "full_rewrite"]


def best_strategy_for_query(strategy_scores: Dict[str, float]) -> str:
    """Returns the strategy with the highest score."""
    if not strategy_scores:
        raise ValueError("Strategy scores dictionary is empty")
    return max(strategy_scores, key=strategy_scores.get)


def entity_density(query: str, entities: List[str]) -> float:
    """Computes density: count of entities in query / total tokens."""
    tokens = query.split()
    if not tokens:
        return 0.0
    entity_tokens = sum(1 for t in tokens if any(e.lower() in t.lower() for e in entities))
    return entity_tokens / len(tokens)


def label_pilot_queries(pilot_queries: List[str], phase1_results: Dict[str, Dict[str, float]]) -> List[int]:
    """
    For pilot queries, label which sub-strategy (0=light_normalize, 1=dual_variant, 2=full_rewrite)
    achieved the highest retrieval metric in Phase 1 results.
    """
    labels = []
    strategy_map = {
        "light_normalize": 0,
        "dual_variant": 1,
        "full_rewrite": 2
    }
    for q in pilot_queries:
        if phase1_results and q in phase1_results:
            scores = phase1_results[q]
            best_strat = max(scores, key=scores.get)
            labels.append(strategy_map.get(best_strat, 0))
        else:
            labels.append(0)
    return labels


def fit_lag_v1(features, labels) -> LogisticRegression:
    """Simple baseline: sklearn LogisticRegression.
    Aligns with newer sklearn versions (no deprecated/removed kwargs).
    """
    X = np.asarray(features)
    y = np.asarray(labels)
    # sklearn LogisticRegression can fit string or integer labels directly
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    return model


def fit_lag_v2(features, labels) -> lgb.LGBMClassifier:
    """Reported model: LightGBM classifier.
    Fits LGBMClassifier on features and labels (handles both int and str labels).
    """
    X = np.asarray(features)
    y = np.asarray(labels)
    model = lgb.LGBMClassifier()
    model.fit(X, y)
    return model


def predict_strategy(cmi: float, lid_entropy: float, entity_density: float, model) -> str:
    """Returns one of SUB_STRATEGIES."""
    features = np.array([[cmi, lid_entropy, entity_density]])
    pred = model.predict(features)[0]
    
    # If the model was trained on integer labels, predict will return an integer.
    # Map it back to the corresponding strategy string.
    if isinstance(pred, (int, np.integer)):
        idx = int(pred)
        if 0 <= idx < len(SUB_STRATEGIES):
            return SUB_STRATEGIES[idx]
        return SUB_STRATEGIES[0]  # Fallback
        
    return str(pred)


def light_normalize(query: str) -> str:
    """Minimal cleanup: lowercase, strip extra whitespace, normalize
    punctuation. Does NOT touch entities or word order. This is the
    'do almost nothing' strategy for queries that don't need heavy
    correction."""
    import re
    if not query:
        return ""
    text = re.sub(r"[\.,;:!\?\(\)\[\]\{\}\"\']", " ", query)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def dual_variant(query: str, entities: List[str] = None) -> List[str]:
    """Returns [query, entity_aware_variant] -- two versions of the
    query to search with and merge results from (e.g. RRF over both).
    entity_aware_variant applies light_normalize AND ensures known
    entities (from the entities list) are surfaced in a consistent
    casing/form, without a full rewrite."""
    import re
    norm = light_normalize(query)
    variant = norm
    if entities:
        for ent in entities:
            pattern = re.compile(re.escape(ent), re.IGNORECASE)
            variant = pattern.sub(ent, variant)
    return [query, variant]


def full_rewrite(
    query: str,
    embed_fn=None,
    translate_hint: bool = True,
    entities: List[str] = None,
    caep_gate=None,
    entity_freq: Dict[str, int] = None,
) -> str:
    """Heaviest correction: apply light_normalize, then hand off to
    CAEP-style entity substitution as a pre-pass (call apply_caep
    internally if entities/gate are available) -- for now, if no
    gate is passed, just apply light_normalize and return. Document
    this as the extension point for a real translation-based rewrite
    later."""
    cleaned = light_normalize(query)
    if entities is not None and caep_gate is not None:
        from setu.operators.caep import apply_caep
        return apply_caep(cleaned, entities, caep_gate, entity_freq=entity_freq or {}, embed_fn=embed_fn)
    return cleaned


def apply_lag(
    query: str,
    strategy: str,
    entities: List[str] = None,
    embed_fn=None,
    caep_gate=None,
    entity_freq: Dict[str, int] = None,
) -> Union[str, List[str]]:
    """Routes to light_normalize / dual_variant / full_rewrite based
    on the strategy label from predict_strategy(). Returns a string
    for light_normalize/full_rewrite, or a list of strings for
    dual_variant."""
    if strategy == "light_normalize":
        return light_normalize(query)
    elif strategy == "dual_variant":
        return dual_variant(query, entities=entities)
    elif strategy == "full_rewrite":
        return full_rewrite(
            query,
            embed_fn=embed_fn,
            translate_hint=True,
            entities=entities,
            caep_gate=caep_gate,
            entity_freq=entity_freq,
        )
    else:
        return light_normalize(query)

