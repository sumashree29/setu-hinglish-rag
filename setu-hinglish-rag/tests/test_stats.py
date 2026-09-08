import numpy as np
from setu.evaluation.stats import paired_wilcoxon, rank_biserial_effect_size, bootstrap_ci, spearman_correlation

def test_wilcoxon_detects_real_difference():
    np.random.seed(0)
    a = np.random.normal(0.5, 0.05, 30)
    b = a + 0.15
    _, p = paired_wilcoxon(a, b)
    assert p < 0.05

def test_spearman_perfect_monotonic():
    x = np.arange(50)
    y = x * 2 + 1
    rho, _ = spearman_correlation(x, y)
    assert rho > 0.99

def test_bootstrap_ci_contains_true_mean():
    np.random.seed(0)
    sample = np.random.normal(0.5, 0.1, 200)
    point, lo, hi = bootstrap_ci(sample, n_resamples=2000)
    assert lo < 0.5 < hi
