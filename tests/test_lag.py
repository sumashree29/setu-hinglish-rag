import numpy as np
from setu.operators.lag import (
    best_strategy_for_query,
    entity_density,
    fit_lag_v1,
    predict_strategy,
    light_normalize,
    dual_variant,
    full_rewrite,
    apply_lag,
)


def test_best_strategy_picks_max():
    assert best_strategy_for_query({"a": 0.4, "b": 0.7, "c": 0.1}) == "b"


def test_entity_density():
    assert entity_density("The RBI regulates all banks.", ["RBI", "SBI"]) == 0.2


def test_light_normalize():
    query = "  What is a BSBDA account...  minimum balance??  "
    norm = light_normalize(query)
    assert norm == "what is a bsbda account minimum balance"
    assert light_normalize("") == ""


def test_dual_variant():
    query = "bsbda account ke saath atm card free milta hai"
    variants = dual_variant(query, entities=["BSBDA", "ATM"])
    assert len(variants) == 2
    assert variants[0] == query
    assert "BSBDA" in variants[1]
    assert "ATM" in variants[1]


def test_full_rewrite_fallback():
    query = "PM-KISAN yojana mein registration kaise karein"
    rewritten = full_rewrite(query, caep_gate=None)
    assert rewritten == "pm-kisan yojana mein registration kaise karein"


def test_apply_lag_dispatch():
    query = "bsbda account details"
    
    # 1. light_normalize
    res_norm = apply_lag(query, strategy="light_normalize")
    assert isinstance(res_norm, str)
    assert res_norm == "bsbda account details"

    # 2. dual_variant
    res_dual = apply_lag(query, strategy="dual_variant", entities=["BSBDA"])
    assert isinstance(res_dual, list)
    assert len(res_dual) == 2
    assert res_dual[0] == query
    assert res_dual[1] == "BSBDA account details"

    # 3. full_rewrite
    res_full = apply_lag(query, strategy="full_rewrite")
    assert isinstance(res_full, str)


def test_fit_and_predict_lag():
    features = [
        [0.1, 0.2, 0.1],  # low CMI -> light_normalize (0)
        [0.4, 0.6, 0.4],  # medium CMI -> dual_variant (1)
        [0.8, 0.9, 0.7],  # high CMI -> full_rewrite (2)
    ]
    labels = [0, 1, 2]
    model = fit_lag_v1(features, labels)
    pred_strat = predict_strategy(0.1, 0.2, 0.1, model)
    assert pred_strat in ["light_normalize", "dual_variant", "full_rewrite"]
