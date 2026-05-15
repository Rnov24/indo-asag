"""Statistical significance testing for model comparison."""

import numpy as np
from scipy import stats


def paired_ttest(scores_a: list[float], scores_b: list[float],
                 alternative: str = "greater") -> dict:
    """One-tailed paired t-test between two sets of CV scores.
    
    Tests H0: scores_a <= scores_b vs H1: scores_a > scores_b.
    
    Args:
        scores_a: Metric scores from model A (per seed or per fold).
        scores_b: Metric scores from model B (per seed or per fold).
        alternative: "greater" (A > B), "less" (A < B), or "two-sided".
        
    Returns:
        Dictionary with t_statistic, p_value, significant (at α=0.05),
        mean_diff, and effect_size (Cohen's d).
    """
    a, b = np.array(scores_a), np.array(scores_b)
    diff = a - b
    
    t_stat, p_val = stats.ttest_rel(a, b, alternative=alternative)
    
    # Cohen's d for paired samples
    d = diff.mean() / diff.std() if diff.std() > 0 else 0.0
    
    return {
        "t_statistic": t_stat,
        "p_value": p_val,
        "significant": p_val < 0.05,
        "mean_diff": diff.mean(),
        "effect_size_d": d,
        "n_pairs": len(a),
    }


def bootstrap_ci(scores: list[float], n_bootstrap: int = 1000,
                 alpha: float = 0.05, seed: int = 42) -> dict:
    """Bootstrap confidence interval for a metric.
    
    Args:
        scores: List of metric values.
        n_bootstrap: Number of bootstrap resamples.
        alpha: Significance level (0.05 for 95% CI).
        seed: Random seed.
        
    Returns:
        Dictionary with mean, ci_lower, ci_upper.
    """
    rng = np.random.RandomState(seed)
    scores = np.array(scores)
    
    boot_means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(scores, size=len(scores), replace=True)
        boot_means.append(sample.mean())
    
    boot_means = np.array(boot_means)
    ci_lower = np.percentile(boot_means, 100 * alpha / 2)
    ci_upper = np.percentile(boot_means, 100 * (1 - alpha / 2))
    
    return {
        "mean": scores.mean(),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "ci_level": 1 - alpha,
    }
