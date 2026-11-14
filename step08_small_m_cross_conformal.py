import numpy as np, pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

alpha, repeats, K, LAM = 0.1, 30, 5, 300
M_LIST = [15, 20, 30, 50]                 # very small numbers of real rare points
X, y = make_classification(n_samples=60000, n_features=20, n_informative=6,
                           weights=[0.98, 0.02], class_sep=0.8, random_state=0)
rare_all, common_all = np.where(y == 1)[0], np.where(y == 0)[0]

def threshold(scores):
    k = int(np.ceil((len(scores) + 1) * (1 - alpha)))
    return np.inf if k > len(scores) else np.sort(scores)[k - 1]

def smote(seeds, n_new, rng, k=5):
    k = min(k, len(seeds) - 1)
    d = np.linalg.norm(seeds[:, None] - seeds[None], axis=2)
    nbrs = np.argsort(d, axis=1)[:, 1:k + 1]
    i = rng.integers(len(seeds), size=n_new)
    j = nbrs[i, rng.integers(nbrs.shape[1], size=n_new)]
    return seeds[i] + rng.random((n_new, 1)) * (seeds[j] - seeds[i])

def fit_model(train_rare, common_train, lam, rng):
    """Forest trained on common points + real rare points + synthetic rare points."""
    synth = smote(X[train_rare], lam, rng) if lam else np.empty((0, X.shape[1]))
    Xtr = np.vstack([X[common_train], X[train_rare], synth])
    ytr = np.concatenate([np.zeros(len(common_train)), np.ones(len(train_rare) + len(synth))])
    return RandomForestClassifier(n_estimators=50, min_samples_leaf=3, n_jobs=-1,
                                  random_state=0).fit(Xtr, ytr)

def split_data(m, seed):
    rng = np.random.default_rng(seed)
    rare = rng.permutation(rare_all); common = rng.permutation(common_all)
    return rng, dict(test_rare=rare[:500], test_com=common[:1000], pool=rare[500:500 + m],
                     cal_com=common[1000:2000], common_train=common[2000:4000])

def summarize(incl_rare, incl_com):
    """incl_* = is that label inside the set, for 500 rare test + 1000 common test points."""
    cov = incl_rare[:500].mean()
    size = ((incl_com[:500].astype(int) + incl_rare[:500]).mean() +
            (incl_com[500:].astype(int) + incl_rare[500:]).mean()) / 2
    return cov, size

# ---- Method 1: dedicated calibration split (what we did in Steps 3-7) ---------------
def run_dedicated(m, n_cal, seed):
    rng, d = split_data(m, seed)
    cal_rare, train_rare = d["pool"][:n_cal], d["pool"][n_cal:]
    model = fit_model(train_rare, d["common_train"], LAM, rng)
    p1 = lambda idx: model.predict_proba(X[idx])[:, 1]
    t_rare = threshold(1 - p1(cal_rare)); t_com = threshold(p1(d["cal_com"]))
    test = np.concatenate([d["test_rare"], d["test_com"]]); p = p1(test)
    return summarize((1 - p) <= t_rare, p <= t_com)

# ---- Method 2: cross-conformal reuse (every rare point trains AND calibrates) -------
def run_cross(m, seed):
    rng, d = split_data(m, seed)
    pool, cal_com = d["pool"], d["cal_com"]
    test = np.concatenate([d["test_rare"], d["test_com"]])
    f_r = np.arange(m) % K                     # fold of each real rare point
    f_c = np.arange(len(cal_com)) % K          # fold of each common calibration point
    P = np.zeros((K, len(test)))               # test-point P(rare) under each fold's model
    s_rare = np.zeros(m); s_com = np.zeros(len(cal_com))
    for k in range(K):
        # model k never sees fold k, and its synthetic points come only from non-fold-k seeds
        model = fit_model(pool[f_r != k], d["common_train"], LAM, rng)
        p1 = lambda idx: model.predict_proba(X[idx])[:, 1]
        P[k] = p1(test)
        s_rare[f_r == k] = 1 - p1(pool[f_r == k])          # score of a rare point under model k
        s_com[f_c == k] = p1(cal_com[f_c == k])            # score of a common point under model k
    # p-value of label "rare" for each test point: how many calibration scores are >= its score
    cnt_r = (s_rare[:, None] >= (1 - P[f_r])).sum(0)
    cnt_c = (s_com[:, None] >= P[f_c]).sum(0)
    incl_rare = (1 + cnt_r) / (m + 1) > alpha
    incl_com = (1 + cnt_c) / (len(cal_com) + 1) > alpha
    return summarize(incl_rare, incl_com)

rows = []
for m in M_LIST:
    methods = {"split 50/50": lambda s: run_dedicated(m, int(round(0.5 * m)), s),
               "split, 9 cal pts": lambda s: run_dedicated(m, 9, s),
               f"cross-conformal K={K}": lambda s: run_cross(m, s)}
    for name, fn in methods.items():
        res = [fn(1000 + r) for r in range(repeats)]        # same splits for every method
        cov = np.array([r[0] for r in res])
        rows.append(dict(m=m, method=name, rare_cov=cov.mean(), worst_cov=cov.min(),
                         cov_sd=cov.std(), reliable=(cov >= 1 - alpha - 0.05).mean(),
                         set_size=np.mean([r[1] for r in res])))
        print("done:", m, name, flush=True)
df = pd.DataFrame(rows)
df.to_csv("step08_results.csv", index=False)
print(df.round(3).to_string(index=False))