"""Real-data regime map (an m-sweep) plus a tie-breaking check.
->  results/real_regime.csv, results/tie_check.csv

  python -m experiments.exp_real_regime                       mammography + ozone-level-8hr (internet needed)
  python -m experiments.exp_real_regime pc1 mammography       choose datasets
  python -m experiments.exp_real_regime --quick               tiny run to check that it works
  python -m experiments.exp_real_regime --jitter              forest only, with ties broken at random (saves real_regime_jitter.csv)

For each dataset the real rare points are subsampled to m = 20, 40, 80 and all available, and rebalancing
(lambda synthetic points) is compared with none, for a forest and a logistic model, on identical splits.
"""
import sys
import numpy as np, pandas as pd
from experiments.common import MODELS, RESULTS, load_dataset, new_split, split_cp


def run(quick=False, datasets=None, jitter=False):
    datasets = datasets or ["mammography", "ozone-level-8hr"]
    reps = 3 if quick else 30
    models = {"forest": MODELS["forest"]} if jitter else MODELS      # ties only affect the forest
    out_name = "real_regime_jitter.csv" if jitter else "real_regime.csv"
    lams = [0, 300] if quick else [0, 100, 300, 1000]
    rows, tie_rows, tie_done = [], [], False
    for name in datasets:
        try:
            X, y = load_dataset(name)
        except Exception as e:
            print(f"real_regime: SKIPPED {name} ({type(e).__name__}: {str(e)[:150]})", flush=True)
            continue
        n_te = max(10, int(0.4 * y.sum())); full = int(y.sum()) - n_te      # test set fixed for every m
        ms = sorted({m for m in ([20, full] if quick else [20, 40, 80, full]) if m <= full})
        for model in models:
            for m in ms:
                res = {lam: [] for lam in lams}
                for r in range(reps):
                    for lam in lams:                                   # same split for every lambda
                        sp, rng = new_split(y, 1000 + r, m, n_te, 4000)
                        res[lam].append(split_cp(X, sp, int(round(0.5 * m)), lam, rng, model, jitter=jitter)["mondrian"])
                base = np.array(res[0])[:, 1]
                for lam in lams:
                    cov, size = np.array(res[lam]).T; diff = size - base
                    rows.append(dict(dataset=name, model=model, m=m, lam=lam, rare_cov=cov.mean(),
                                     cov_sd=cov.std(), set_size=size.mean(), size_vs_lam0=diff.mean(),
                                     ci95=1.96 * diff.std() / np.sqrt(len(diff))))
                print("real_regime:", name, model, "m =", m, flush=True)
                pd.DataFrame(rows).to_csv(RESULTS / out_name, index=False)

        if not tie_done and not jitter:                      # tie check on the first dataset that loaded (forest, all rare points)
            tie_done = True
            for jit in (False, True):
                res = {0: [], 300: []}
                for r in range(reps):
                    for lam in res:
                        sp, rng = new_split(y, 1000 + r, full, n_te, 4000)
                        res[lam].append(split_cp(X, sp, int(round(0.5 * full)), lam, rng, "forest",
                                                 jitter=jit)["mondrian"])
                base = np.array(res[0])[:, 1]
                for lam, v in res.items():
                    cov, size = np.array(v).T; diff = size - base
                    tie_rows.append(dict(dataset=name, jitter=jit, lam=lam, rare_cov=cov.mean(),
                                         set_size=size.mean(), size_vs_lam0=diff.mean(),
                                         ci95=1.96 * diff.std() / np.sqrt(len(diff))))
            pd.DataFrame(tie_rows).to_csv(RESULTS / "tie_check.csv", index=False)
            print("tie check done on", name, flush=True)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    run("--quick" in sys.argv, args or None, jitter="--jitter" in sys.argv)
