"""RQ4: does the derived rule beat simple splits? (Step 7)  ->  results/rule_vs_baselines.csv"""
import sys
import numpy as np, pandas as pd
from allocation.rule import rule_n_cal
from experiments.common import ALPHA, LAM, RESULTS, synthetic_data, new_split, split_cp

DELTA = 0.05


def run(quick=False):
    X, y = synthetic_data()
    reps = 3 if quick else 30
    ms = [30, 60] if quick else [30, 60, 120]
    rows = []
    for m in ms:
        n_rule = rule_n_cal(m, ALPHA, DELTA)
        methods = {"A uniform 50/50": (int(round(0.5 * m)), 0),
                   "B conventional 80/20": (int(round(0.2 * m)), 0),
                   "C rule": (n_rule, 0),
                   "D rule + rebalancing": (n_rule, LAM),
                   "E uniform 50/50 + rebalancing": (int(round(0.5 * m)), LAM)}
        out = {}
        for name, (n_cal, lam) in methods.items():
            res = []
            for r in range(reps):                       # same splits for every method
                sp, rng = new_split(y, 1000 + r, m)
                res.append(split_cp(X, sp, n_cal, lam, rng, "forest")["mondrian"])
            out[name] = (n_cal, np.array(res))
            print("rule:", m, name, flush=True)
        base = out["A uniform 50/50"][1][:, 1]
        for name, (n_cal, arr) in out.items():
            cov, size = arr[:, 0], arr[:, 1]; diff = size - base
            rows.append(dict(m=m, method=name, n_cal=n_cal, rare_cov=cov.mean(),
                             reliable=(cov >= 1 - ALPHA - DELTA).mean(), set_size=size.mean(),
                             size_vs_A=diff.mean(), ci95=1.96 * diff.std() / np.sqrt(len(diff))))
    pd.DataFrame(rows).to_csv(RESULTS / "rule_vs_baselines.csv", index=False)


if __name__ == "__main__":
    run("--quick" in sys.argv)
