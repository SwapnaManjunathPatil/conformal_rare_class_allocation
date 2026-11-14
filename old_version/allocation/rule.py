"""The calibration-size rule (Step 7)."""
import numpy as np
from scipy.stats import beta
from conformal.core import slack


def rule_n_cal(m, alpha=0.1, delta=0.05, eta=0.10):
    """Number of real rare points to keep for calibration, or None if m is too small.

    1. Reliability: with n points, coverage ~ Beta(n+1-l, l), l = floor((n+1)*alpha).
       Need P(coverage >= 1 - alpha - delta) >= 1 - eta.
    2. Efficiency: n points really use level ceil((n+1)(1-alpha))/(n+1); pick the size in the
       next 1/alpha candidates with the least over-coverage (smallest n on ties).
    3. Learning curve: more training points never hurt, so no more calibration than needed.
    """
    for n0 in range(int(np.ceil(1 / alpha - 1)), m):
        l = int(np.floor((n0 + 1) * alpha))
        if 1 - beta.cdf(1 - alpha - delta, n0 + 1 - l, l) >= 1 - eta:
            window = range(n0, min(n0 + int(1 / alpha), m))
            return min(window, key=lambda n: (round(slack(n, alpha), 6), n))
    return None
