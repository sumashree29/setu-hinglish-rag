"""Tests for setu/diagnosis/cmi.py. OWNER: R1."""
from setu.diagnosis.cmi import cmi


def test_pure_english_query_has_low_cmi():
    score = cmi("what is my loan status")
    assert score == 0.0 or score < 0.1


def test_heavily_mixed_query_has_high_cmi():
    score = cmi("mera loan status kaise pata karu")
    assert score > 0.3


def test_empty_query_returns_zero():
    assert cmi("") == 0.0
