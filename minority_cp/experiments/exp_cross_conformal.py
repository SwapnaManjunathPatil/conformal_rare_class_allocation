"""RQ5: at very small m, reuse (cross-conformal) vs a dedicated split? (Step 8)
->  results/cross_conformal.csv"""
import sys
import numpy as np, pandas as pd
from experiments.common import ALPHA, K, LAM, RESULTS, synthetic_data, new_split, split_cp, cross_cp


def run(quick=False):
    X, y = synthetic_data()
    reps = 3 if quick else 30
    ms = [15, 30] if quick else [15, 20, 30, 50]
    rows = []
    for m in ms:
        methods = {"split 50/50": lambda sp, rng: split_cp(X, sp, int(round(0.5 * m)), LAM, rng, "forest")["mondrian"],
                   "split, 9 cal points": lambda sp, rng: split_cp(X, sp, 9, LAM, rng, "forest")["mondrian"],
                   f"cross-conformal K={K}": lambda sp, rng: cross_cp(X, sp, LAM, rng, "forest")}
        for name, fn in methods.items():
            res = []
            for r in range(reps):
                sp, rng = new_split(y, 1000 + r, m)
                res.append(fn(sp, rng))
            cov, size = np.array(res).T
            rows.append(dict(m=m, method=name, rare_cov=cov.mean(), worst_cov=cov.min(),
                             cov_sd=cov.std(), reliable=(cov >= 1 - ALPHA - 0.05).mean(),
                             set_size=size.mean()))
            print("cross:", m, name, flush=True)
    pd.DataFrame(rows).to_csv(RESULTS / "cross_conformal.csv", index=False)


if __name__ == "__main__":
    run("--quick" in sys.argv)
