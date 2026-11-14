"""Worked example: apply the plan to your own data.   python examples/use_on_your_data.py

Replace the two lines marked YOUR DATA with your own X (features) and y (1 = rare class, 0 = other).
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np
from allocation.api import RareClassConformal
from experiments.common import load_dataset

X, y = load_dataset("digits9")                       # YOUR DATA: replace with your X, y
rng = np.random.default_rng(1)

# Always keep a final test set that the method never sees (here: 40% of the rare points, 25% of the rest)
rare, common = rng.permutation(np.where(y == 1)[0]), rng.permutation(np.where(y == 0)[0])
test = np.concatenate([rare[:20], common[:len(common) // 4]])
fit_idx = np.setdiff1d(np.arange(len(y)), test)

cp = RareClassConformal(alpha=0.1).fit(X[fit_idx], y[fit_idx])
print(cp.plan_, "\n")

sets = cp.predict_sets(X[test])                      # column 0 = "common" in set, column 1 = "rare" in set
is_rare = y[test] == 1
print("rare-class coverage  :", sets[is_rare, 1].mean().round(3), "(target 0.90)")
print("common-class coverage:", sets[~is_rare, 0].mean().round(3), "(target 0.90)")
print("average set size     :", sets.sum(1).mean().round(3), "labels per case (1 is ideal, 2 is useless)")
