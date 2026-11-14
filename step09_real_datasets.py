import sys
import numpy as np, pandas as pd
from scipy.stats import beta
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

alpha, repeats, K, LAM = 0.1, 30, 5, 300
DELTA, ETA = 0.05, 0.10
# run:  python step09_real_datasets.py                     (two built-in datasets, no internet)
#       python step09_real_datasets.py mammography yeast_me2   (needs imbalanced-learn + internet)
DATASETS = sys.argv[1:] or ["digits9", "cancer_imb"]

def load_dataset(name):
    """Return (X, y) with y = 1 for the minority class."""
    rng = np.random.default_rng(0)
    if name in ("digits9", "cancer_imb"):
        # real data from scikit-learn; the minority class is thinned out to create imbalance
        if name == "digits9":
            from sklearn.datasets import load_digits
            d = load_digits(); lab = (d.target == 9).astype(int); n_keep = 50
        else:
            from sklearn.datasets import load_breast_cancer
            d = load_breast_cancer(); lab = (d.target == 0).astype(int); n_keep = 40  # malignant
        mino = rng.choice(np.where(lab == 1)[0], n_keep, replace=False)
        keep = np.concatenate([np.where(lab == 0)[0], mino])
        X, y = d.data[keep], lab[keep]
    else:
        from imblearn.datasets import fetch_datasets   # downloads the data on first use
        d = fetch_datasets()[name]
        X, y = d.data, (d.target == 1).astype(int)
    # note: scaling uses all rows (features only, no labels); fine for a first study
    return StandardScaler().fit_transform(X), y

def threshold(scores):
    k = int(np.ceil((len(scores) + 1) * (1 - alpha)))
    return np.inf if k > len(scores) else np.sort(scores)[k - 1]

def smote(seeds, n_new, rng, k=5):
    if len(seeds) < 3:
        return np.empty((0, seeds.shape[1]))
    k = min(k, len(seeds) - 1)
    d = np.linalg.norm(seeds[:, None] - seeds[None], axis=2)
    nbrs = np.argsort(d, axis=1)[:, 1:k + 1]
    i = rng.integers(len(seeds), size=n_new)
    j = nbrs[i, rng.integers(nbrs.shape[1], size=n_new)]
    return seeds[i] + rng.random((n_new, 1)) * (seeds[j] - seeds[i])

def slack(n):
    return np.ceil((n + 1) * (1 - alpha)) / (n + 1) - (1 - alpha)

def rule_n_cal(m):
    """Step 7 rule: smallest calibration size with reliable coverage and little over-coverage."""
    for n0 in range(int(np.ceil(1 / alpha - 1)), m):
        l = int(np.floor((n0 + 1) * alpha))
        if 1 - beta.cdf(1 - alpha - DELTA, n0 + 1 - l, l) >= 1 - ETA:
            window = range(n0, min(n0 + int(1 / alpha), m))
            return min(window, key=lambda n: (round(slack(n), 6), n))
    return None

def run_once(X, y, seed):
    rng = np.random.default_rng(seed)
    rare = rng.permutation(np.where(y == 1)[0]); common = rng.permutation(np.where(y == 0)[0])
    n_te = max(10, int(0.4 * len(rare)))                 # 40% of rare points held out for testing
    test_rare, pool = rare[:n_te], rare[n_te:]; m = len(pool)   # pool = our budget of m real rare points
    nt = min(1000, len(common) // 4)
    test_com, cal_com = common[:nt], common[nt:2 * nt]
    train_com = common[2 * nt: 2 * nt + 4000]
    test = np.concatenate([test_rare, test_com]); nr = len(test_rare)

    def fit(train_rare, lam):
        synth = smote(X[train_rare], lam, rng) if lam else np.empty((0, X.shape[1]))
        Xtr = np.vstack([X[train_com], X[train_rare], synth])
        ytr = np.concatenate([np.zeros(len(train_com)), np.ones(len(train_rare) + len(synth))])
        mod = RandomForestClassifier(n_estimators=50, min_samples_leaf=3, n_jobs=-1,
                                     random_state=0).fit(Xtr, ytr)
        return lambda idx: mod.predict_proba(X[idx])[:, 1]

    def summarize(incl_rare, incl_com):
        cov = incl_rare[:nr].mean()
        size = ((incl_com[:nr].astype(int) + incl_rare[:nr]).mean() +
                (incl_com[nr:].astype(int) + incl_rare[nr:]).mean()) / 2
        return cov, size

    def split_cp(n_cal, lam):
        p1 = fit(pool[n_cal:], lam); p = p1(test)
        t_r = threshold(1 - p1(pool[:n_cal])); t_c = threshold(p1(cal_com))
        return summarize((1 - p) <= t_r, p <= t_c), p1, p, n_cal

    out = {}
    n_half = int(round(0.5 * m))
    (out["Mondrian 50/50, plain"]), p1, p, _ = split_cp(n_half, 0)
    q = threshold(np.concatenate([1 - p1(pool[:n_half]), p1(cal_com)]))     # one pooled threshold
    out["Marginal 50/50, plain"] = summarize((1 - p) <= q, p <= q)
    out["Mondrian 50/50 + rebal."] = split_cp(n_half, LAM)[0]
    n_rule = rule_n_cal(m)
    if n_rule is not None and m - n_rule >= 5:
        out["Rule split + rebal."] = split_cp(n_rule, LAM)[0]

    # cross-conformal: every rare point trains K-1 models and calibrates 1
    f_r = np.arange(m) % K; f_c = np.arange(len(cal_com)) % K
    P = np.zeros((K, len(test))); s_r = np.zeros(m); s_c = np.zeros(len(cal_com))
    for k in range(K):
        pk = fit(pool[f_r != k], LAM)
        P[k] = pk(test); s_r[f_r == k] = 1 - pk(pool[f_r == k]); s_c[f_c == k] = pk(cal_com[f_c == k])
    in_r = (1 + (s_r[:, None] >= (1 - P[f_r])).sum(0)) / (m + 1) > alpha
    in_c = (1 + (s_c[:, None] >= P[f_c]).sum(0)) / (len(cal_com) + 1) > alpha
    out["Cross-conformal + rebal."] = summarize(in_r, in_c)
    return out, m

rows = []
for name in DATASETS:
    X, y = load_dataset(name)
    res, m = {}, None
    for r in range(repeats):                     # same splits for every method
        o, m = run_once(X, y, 1000 + r)
        for k, v in o.items():
            res.setdefault(k, []).append(v)
    base = np.array([v[1] for v in res["Mondrian 50/50, plain"]])
    print(f"\n=== {name}: {len(y)} rows, {int(y.sum())} minority ({100*y.mean():.1f}%), "
          f"budget m = {m} real rare points ===")
    for meth, vals in res.items():
        cov = np.array([v[0] for v in vals]); size = np.array([v[1] for v in vals])
        diff = size - base
        rows.append(dict(dataset=name, m=m, method=meth, rare_cov=cov.mean(), cov_sd=cov.std(),
                         worst_cov=cov.min(), set_size=size.mean(),
                         size_vs_plain=diff.mean(), ci95=1.96 * diff.std() / np.sqrt(len(diff))))
    print(pd.DataFrame(rows[-len(res):]).drop(columns=["dataset", "m"]).round(3).to_string(index=False))
pd.DataFrame(rows).to_csv("step09_results.csv", index=False)
print("\nsaved step09_results.csv")