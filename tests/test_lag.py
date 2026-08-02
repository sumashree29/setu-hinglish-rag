import numpy as np
from setu.operators.lag import best_strategy_for_query, fit_lag_v1, predict_strategy, entity_density

def test_best_strategy_picks_max():
    assert best_strategy_for_query({"a": 0.4, "b": 0.7, "c": 0.1}) == "b"

def test_entity_density():
    assert entity_density("The RBI regulates all banks.", ["RBI", "SBI"]) == 0.2
