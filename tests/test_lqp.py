import numpy as np
from setu.operators.lqp import fit_lqp, apply_lqp, apply_lqp_batch

def test_lqp_boundaries():
    np.random.seed(0)
    d, n = 16, 200
    true_W = np.random.randn(d, d) * 0.3 + np.eye(d)
    X = np.random.randn(n, d)
    Y = X @ true_W + np.random.randn(n, d) * 0.01

    model = fit_lqp(X, Y, alpha_reg=0.1)
    test_query = X[0]

    # CMI=0 -> identity (unchanged)
    projected_zero = apply_lqp(test_query, cmi_score=0.0, model=model, cmi_max=50.0)
    assert np.linalg.norm(projected_zero - test_query) < 1e-8

    # CMI=max -> full projection
    projected_full = apply_lqp(test_query, cmi_score=50.0, model=model, cmi_max=50.0)
    expected = test_query @ model.coef_.T
    assert np.linalg.norm(projected_full - expected) < 1e-8
