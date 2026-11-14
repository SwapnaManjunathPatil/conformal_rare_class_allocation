"""Checks for the advisor and the user-facing class.  python tests/test_advisor.py"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from sklearn.datasets import make_classification
from allocation.advisor import recommend, leakage_safe_split, check_no_leakage
from allocation.api import RareClassConformal

assert not recommend(5).feasible and not recommend(15, alpha=0.05).feasible
assert recommend(30).strategy == "cross_conformal"
p = recommend(60); assert p.strategy == "dedicated_split" and p.n_cal == 19 and p.n_train == 41

tr, cal = leakage_safe_split(np.arange(40), 16, np.random.default_rng(0))
assert len(cal) == 16 and check_no_leakage(tr, cal)
try:
    check_no_leakage(tr, cal, seed_idx=cal[:3]); raise SystemExit("leak not caught")
except ValueError:
    pass

# end-to-end: coverage on fresh data should be near 0.9 for both strategies
X, y = make_classification(n_samples=20000, n_features=10, n_informative=6, weights=[0.97, 0.03],
                           class_sep=1.5, random_state=0)
for strat in ("split", "cross"):
    covs = []
    for seed in range(8):
        rng = np.random.default_rng(seed)
        rare, com = rng.permutation(np.where(y == 1)[0]), rng.permutation(np.where(y == 0)[0])
        fit = np.concatenate([rare[:60], com[:3000]]); test = np.concatenate([rare[60:260], com[3000:4000]])
        cp = RareClassConformal(model="logistic", n_synthetic=0, strategy=strat, seed=seed).fit(X[fit], y[fit])
        s = cp.predict_sets(X[test]); covs.append(s[y[test] == 1, 1].mean())
    assert 0.82 < np.mean(covs) < 0.99, (strat, np.mean(covs))
print("all advisor tests passed")
