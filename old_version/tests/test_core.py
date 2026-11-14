"""Quick sanity checks.  Run:  python tests/test_core.py   (from the project root)"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from conformal.core import threshold, min_calibration_size, slack
from allocation.rule import rule_n_cal

assert min_calibration_size(0.1) == 9 and min_calibration_size(0.05) == 19
assert np.isinf(threshold(np.random.rand(8), 0.1))           # 8 points: too few
assert np.isfinite(threshold(np.random.rand(9), 0.1))        # 9 points: just enough
assert abs(slack(19, 0.1)) < 1e-12 and slack(15, 0.1) > 0.03
assert rule_n_cal(60) == 19 and rule_n_cal(15) is None

rng = np.random.default_rng(0)                               # coverage ~ 1 - alpha on average
cov = [(rng.random(5000) <= threshold(rng.random(30), 0.1)).mean() for _ in range(500)]
assert 0.89 < np.mean(cov) < 0.93, np.mean(cov)
print("all tests passed")
