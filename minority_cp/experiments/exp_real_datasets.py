"""All methods on real datasets (Step 9)  ->  results/real_datasets.csv

  python -m experiments.exp_real_datasets                        built-in sets, no internet
  python -m experiments.exp_real_datasets mammography yeast_me2  needs imbalanced-learn + internet
"""
import sys
import numpy as np, pandas as pd
from allocation.rule import rule_n_cal
from experiments.common import ALPHA, LAM, RESULTS, load_dataset, new_split, split_cp, cross_cp


def run(quick=False, datasets=None):
    datasets = datasets or ["digits9", "cancer_imb"]
    reps = 3 if quick else 30
    rows = []
    for name in datasets:
        try:
            X, y = load_dataset(name)
        except Exception as e:                 # a failed download should not lose the other datasets
            print(f"real: SKIPPED {name} ({type(e).__name__}: {str(e)[:150]})", flush=True)
            continue
        n_te = max(10, int(0.4 * y.sum()))               # 40% of rare points held out for testing
        res = {}; m = None
        for r in range(reps):
            sp, rng = new_split(y, 1000 + r, None, n_te, 4000)
            m = len(sp["pool"]); n_half = int(round(0.5 * m))
            plain = split_cp(X, sp, n_half, 0, rng, "forest")
            res.setdefault("Mondrian 50/50, plain", []).append(plain["mondrian"])
            res.setdefault("Marginal 50/50, plain", []).append(plain["marginal"])
            res.setdefault("Mondrian 50/50 + rebal.", []).append(
                split_cp(X, sp, n_half, LAM, rng, "forest")["mondrian"])
            n_rule = rule_n_cal(m, ALPHA)
            if n_rule is not None and m - n_rule >= 5:
                res.setdefault("Rule split + rebal.", []).append(
                    split_cp(X, sp, n_rule, LAM, rng, "forest")["mondrian"])
            res.setdefault("Cross-conformal + rebal.", []).append(cross_cp(X, sp, LAM, rng, "forest"))
        base = np.array(res["Mondrian 50/50, plain"])[:, 1]
        for meth, vals in res.items():
            cov, size = np.array(vals).T; diff = size - base
            rows.append(dict(dataset=name, m=m, method=meth, rare_cov=cov.mean(), cov_sd=cov.std(),
                             worst_cov=cov.min(), set_size=size.mean(), size_vs_plain=diff.mean(),
                             ci95=1.96 * diff.std() / np.sqrt(len(diff))))
        print("real:", name, f"({len(y)} rows, {int(y.sum())} minority, m = {m})", flush=True)
        pd.DataFrame(rows).to_csv(RESULTS / "real_datasets.csv", index=False)   # saved after every dataset


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    run("--quick" in sys.argv, args or None)
