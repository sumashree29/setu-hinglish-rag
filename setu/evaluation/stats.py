"""
Statistical testing protocol — applied uniformly across H1-H10.
OWNER: R3 | PHASE: 5 (plan §7.4)
"""
from typing import List, Tuple, Dict
import numpy as np
from scipy import stats


def paired_wilcoxon(metric_a: List[float], metric_b: List[float]) -> Tuple[float, float]:
    """
    Perform a paired Wilcoxon signed-rank test.
    Returns (statistic, p_value).
    """
    a = np.asarray(metric_a)
    b = np.asarray(metric_b)
    if np.array_equal(a, b):
        return 0.0, 1.0
    res = stats.wilcoxon(a, b)
    return float(res.statistic), float(res.pvalue)


def rank_biserial_effect_size(metric_a: List[float], metric_b: List[float]) -> float:
    """
    Compute the matched-pairs rank-biserial correlation coefficient.
    
    WARNING: The sign of the rank-biserial correlation can vary depending on
    the convention of positive/negative differences. Always print and check
    the raw means of both groups alongside it to confirm which group won, rather
    than trusting the sign alone. The magnitude represents how consistent the
    difference is.
    
    Returns a value in [-1.0, 1.0].
    """
    a = np.asarray(metric_a)
    b = np.asarray(metric_b)
    diff = b - a
    non_zero_diff = diff[diff != 0]
    if len(non_zero_diff) == 0:
        return 0.0

    ranks = stats.rankdata(np.abs(non_zero_diff))
    w_plus = np.sum(ranks[non_zero_diff > 0])
    w_minus = np.sum(ranks[non_zero_diff < 0])
    total_ranks = np.sum(ranks)

    if total_ranks == 0:
        return 0.0

    return float((w_plus - w_minus) / total_ranks)


def bootstrap_ci(
    metric_values: List[float],
    n_resamples: int = 10000,
    confidence: float = 0.95
) -> Tuple[float, float, float]:
    """
    Compute a bootstrap confidence interval for the mean.
    Returns (point_estimate_mean, low_ci, high_ci).
    """
    arr = np.asarray(metric_values)
    point_estimate = float(np.mean(arr))
    if len(arr) == 0 or np.all(arr == arr[0]):
        return point_estimate, point_estimate, point_estimate
    
    # Use percentile method or BCa for robust interval estimation
    try:
        res = stats.bootstrap(
            (arr,),
            np.mean,
            confidence_level=confidence,
            n_resamples=min(n_resamples, 2000),
            method='BCa'
        )
        return point_estimate, float(res.confidence_interval.low), float(res.confidence_interval.high)
    except Exception:
        res = stats.bootstrap(
            (arr,),
            np.mean,
            confidence_level=confidence,
            n_resamples=min(n_resamples, 2000),
            method='percentile'
        )
        return point_estimate, float(res.confidence_interval.low), float(res.confidence_interval.high)


def spearman_correlation(x: List[float], y: List[float]) -> Tuple[float, float]:
    """
    Used for the confidence-signal correlation study (plan §5.2, H10).
    Returns (rho, p_value).
    """
    res = stats.spearmanr(x, y)
    return float(res.statistic), float(res.pvalue)
