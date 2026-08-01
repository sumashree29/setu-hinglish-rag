"""
Statistical testing protocol — applied uniformly across H1-H10.
OWNER: R3 | PHASE: 5 (plan §7.4)
"""
from typing import List, Tuple
from scipy import stats


def paired_wilcoxon(metric_a: List[float], metric_b: List[float]) -> Tuple[float, float]:
    """Returns (statistic, p_value). TODO (R3): scipy.stats.wilcoxon(metric_a, metric_b)."""
    raise NotImplementedError("R3: implement paired Wilcoxon signed-rank test")


def rank_biserial_effect_size(metric_a: List[float], metric_b: List[float]) -> float:
    """TODO (R3): compute rank-biserial correlation as the effect size to report
    alongside the Wilcoxon p-value."""
    raise NotImplementedError("R3: implement effect size")


def bootstrap_ci(metric_values: List[float], n_resamples: int = 10000, confidence: float = 0.95):
    """TODO (R3): scipy.stats.bootstrap — built into modern SciPy, no extra dep."""
    raise NotImplementedError("R3: implement bootstrap confidence interval")


def spearman_correlation(x: List[float], y: List[float]) -> Tuple[float, float]:
    """Used for the confidence-signal correlation study (plan §5.2, H10).
    Returns (rho, p_value). TODO (R3): scipy.stats.spearmanr."""
    raise NotImplementedError("R3: implement Spearman correlation")
