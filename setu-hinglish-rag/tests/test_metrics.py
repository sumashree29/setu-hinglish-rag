"""Tests for setu/evaluation/metrics.py. OWNER: R1/R3."""
import math
from setu.evaluation.metrics import confidence_proxy


def test_confidence_proxy_margin():
    scores = [10.0, 8.5, 5.0, 1.0]
    # margin should be sorted_scores[0] - sorted_scores[1] = 10.0 - 8.5 = 1.5
    assert math.isclose(confidence_proxy(scores, method="margin"), 1.5)


def test_confidence_proxy_entropy():
    scores = [0.5, 0.5, 0.5, 0.5]
    # entropy should be log2(4) = 2.0
    assert math.isclose(confidence_proxy(scores, method="entropy"), 2.0)


def test_confidence_proxy_entropy_negatives():
    scores = [-0.5, 0.5, -0.5, 0.5]
    # negative scores should run successfully through softmax fallback and return positive entropy
    ent = confidence_proxy(scores, method="entropy")
    assert ent > 0.0
