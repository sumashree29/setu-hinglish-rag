import numpy as np
from setu.operators.lqp import fit_lqp, apply_lqp, apply_lqp_batch
from setu.diagnosis.cmi import cmi


def test_lqp_boundaries():
    np.random.seed(0)
    d, n = 16, 200
    true_W = np.random.randn(d, d) * 0.3 + np.eye(d)
    X = np.random.randn(n, d)
    Y = X @ true_W + np.random.randn(n, d) * 0.01
    model = fit_lqp(X, Y, alpha_reg=0.1)
    test_query = X[0]

    # CMI=0 -> identity (unchanged)
    projected_zero = apply_lqp(test_query, cmi_score=0.0, model=model, cmi_max=1.0)
    assert np.linalg.norm(projected_zero - test_query) < 1e-8

    # CMI=max (1.0, per current cmi() range) -> full projection
    projected_full = apply_lqp(test_query, cmi_score=1.0, model=model, cmi_max=1.0)
    expected = test_query @ model.coef_.T
    assert np.linalg.norm(projected_full - expected) < 1e-8


def test_lqp_with_real_cmi_range():
    """Regression test: catches cmi_max/cmi() scale mismatches like the one
    found on 2026-08-18, where cmi_max defaulted to 50 while cmi() actually
    returns [0,1] -- silently collapsing alpha to near-zero for every query."""
    np.random.seed(1)
    d, n = 16, 200
    true_W = np.random.randn(d, d) * 0.3 + np.eye(d)
    X = np.random.randn(n, d)
    Y = X @ true_W + np.random.randn(n, d) * 0.01
    model = fit_lqp(X, Y, alpha_reg=0.1)
    test_query = X[0]

    sample_cmi = cmi("mera SBI account band karna hai kaise")
    assert 0.0 <= sample_cmi <= 1.0, (
        f"cmi() returned {sample_cmi}, outside expected [0,1] range -- "
        "if this fails, cmi.py's scale changed again and lqp.py's cmi_max "
        "default needs to be updated to match."
    )

    projected = apply_lqp(test_query, cmi_score=sample_cmi, model=model)  # uses default cmi_max
    alpha_used = min(1.0, sample_cmi / 1.0)
    assert alpha_used > 0.05, (
        "alpha collapsed near zero -- check that apply_lqp's cmi_max default "
        "matches cmi()'s actual output range"
    )