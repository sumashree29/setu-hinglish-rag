"""Tests for setu/evaluation/metrics.py. OWNER: R1/R3."""
import math
from setu.evaluation.metrics import recall_at_k, mrr, confidence_proxy


def test_recall_at_k_perfect_match():
    # TODO: assert recall_at_k(["a","b","c"], ["a"], k=3) == 1.0
    pass


def test_mrr_first_position():
    # TODO: assert mrr(["a","b"], ["a"]) == 1.0
    pass


def test_confidence_proxy_margin():
    scores = [10.0, 8.5, 5.0, 1.0]
    # margin should be sorted_scores[0] - sorted_scores[1] = 10.0 - 8.5 = 1.5
    assert math.isclose(confidence_proxy(scores, method="margin"), 1.5)


def test_confidence_proxy_entropy():
    scores = [0.5, 0.5, 0.5, 0.5]
    # entropy should be log2(4) = 2.0
    assert math.isclose(confidence_proxy(scores, method="entropy"), 2.0)
