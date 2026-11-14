"""RQ2: when does rebalancing shrink or inflate the sets? (Steps 5-6)  ->  results/regime_map.csv"""
import sys
import numpy as np, pandas as pd
from experiments.common import MODELS, RESULTS, synthetic_data, new_split, split_cp


def run(quick=False):
    X, y = synthetic_data()
    reps = 3 if quick else 30
    ms = [20, 100] if quick else [20, 40, 100, 200]
    lams = [0, 300] if quick else [0, 100, 300, 1000]
    rows = []
    for model in MODELS:
        for m in ms:
            for lam in lams:
                res = []
                for r in range(reps):
                    sp, rng = new_split(y, 1000 + r, m)
                    res.append(split_cp(X, sp, int(round(0.5 * m)), lam, rng, model)["mondrian"])
                cov, size = np.array(res).T
                rows.append(dict(model=model, m=m, lam=lam, rare_cov=cov.mean(),
                                 cov_sd=cov.std(), set_size=size.mean()))
                print("regime_map:", model, m, lam, flush=True)
    pd.DataFrame(rows).to_csv(RESULTS / "regime_map.csv", index=False)


if __name__ == "__main__":
    run("--quick" in sys.argv)
