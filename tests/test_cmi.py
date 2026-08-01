"""Tests for setu/diagnosis/cmi.py. OWNER: R1.
Fill these in as you implement cmi() — start with obvious cases."""
from setu.diagnosis.cmi import cmi


def test_pure_english_query_has_low_cmi():
    # TODO: assert cmi("what is my loan status") is close to 0.0
    pass


def test_heavily_mixed_query_has_high_cmi():
    # TODO: assert cmi("mera loan status kaise pata karu") is meaningfully > 0
    pass


def test_empty_query_returns_zero():
    # TODO: assert cmi("") == 0.0
    pass
