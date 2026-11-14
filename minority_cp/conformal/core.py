"""Basic split-conformal building blocks (Steps 1-3)."""
import numpy as np


def threshold(scores, alpha):
    """Finite-sample corrected quantile of calibration scores.
    Returns +inf when there are too few points (the set then contains every label)."""
    n = len(scores)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    return np.inf if k > n else np.sort(scores)[k - 1]


def min_calibration_size(alpha):
    """Smallest per-class calibration size that gives a finite threshold: n >= 1/alpha - 1."""
    return int(np.ceil(1 / alpha - 1))


def slack(n, alpha):
    """Over-coverage: the quantile level actually used, ceil((n+1)(1-alpha))/(n+1), minus 1-alpha."""
    return np.ceil((n + 1) * (1 - alpha)) / (n + 1) - (1 - alpha)
