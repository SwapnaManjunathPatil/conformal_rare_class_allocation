"""RQ1: how small can the rare calibration set be? (Steps 1-2)  ->  results/min_size.csv"""
import sys
import numpy as np, pandas as pd
from conformal.core import threshold, min_calibration_size
from experiments.common import ALPHA, RESULTS


def run(quick=False):
    reps = 300 if quick else 3000
    rng = np.random.default_rng(0)
    rows = []
    for n in list(range(3, 31)) + [40, 50, 60, 80, 100]:
        # With uniform scores, coverage of a threshold equals the threshold value itself.
        t = np.array([threshold(rng.random(n), ALPHA) for _ in range(reps)])
        ok = np.isfinite(t)
        l = int(np.floor((n + 1) * ALPHA))
        theory = np.sqrt((n + 1 - l) * l / ((n + 1) ** 2 * (n + 2))) if l > 0 else np.nan
        rows.append(dict(n_cal=n, finite_share=ok.mean(),
                         cov_mean=t[ok].mean() if ok.any() else np.nan,
                         cov_sd=t[ok].std() if ok.any() else np.nan, theory_sd=theory))
    pd.DataFrame(rows).to_csv(RESULTS / "min_size.csv", index=False)
    print(f"min_size: minimum calibration size for alpha={ALPHA} is {min_calibration_size(ALPHA)}")


if __name__ == "__main__":
    run("--quick" in sys.argv)
