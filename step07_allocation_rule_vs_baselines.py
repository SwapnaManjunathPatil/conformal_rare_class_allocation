import numpy as np, pandas as pd
from scipy.stats import beta
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

alpha, repeats = 0.1, 30
DELTA, ETA = 0.05, 0.10          # want: coverage >= 1 - alpha - DELTA with probability >= 1 - ETA
M_LIST = [30, 60, 120]
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

# ---- THE RULE: smallest calibration size that gives reliable coverage -------------
def slack(n):
    """Over-coverage: the quantile level actually used minus the target 1 - alpha."""
    return np.ceil((n + 1) * (1 - alpha)) / (n + 1) - (1 - alpha)

def rule_n_cal(m):
    """Term 1 (finite-sample coverage): coverage of split CP with n points follows
    Beta(n+1-l, l), l = floor((n+1)*alpha). Need P(coverage >= 1-alpha-DELTA) >= 1-ETA.
    Term 2 (efficiency): n points really use level ceil((n+1)(1-alpha))/(n+1), which
    over-covers (bigger sets) unless n is 'tight'. Term 3 (learning curve): more training
    points never hurt. So: among the next 1/alpha sizes after the reliability limit,
    take the smallest n with the least over-coverage."""
    for n0 in range(int(np.ceil(1 / alpha - 1)), m):
        l = int(np.floor((n0 + 1) * alpha))
        if 1 - beta.cdf(1 - alpha - DELTA, n0 + 1 - l, l) >= 1 - ETA:
            window = range(n0, min(n0 + int(1 / alpha), m))
            return min(window, key=lambda n: (round(slack(n), 6), n))
    return None                   # m too small: no valid allocation exists

def one_run(m, n_cal, lam, seed):
    rng = np.random.default_rng(seed)
    rare = rng.permutation(rare_all); common = rng.permutation(common_all)
    test_rare, test_com = rare[:500], common[:1000]
    pool = rare[500:500 + m]
    cal_rare, train_rare = pool[:n_cal], pool[n_cal:]
    cal_com = common[1000:2000]
    synth = smote(X[train_rare], lam, rng) if lam else np.empty((0, X.shape[1]))
    Xtr = np.vstack([X[common[2000:4000]], X[train_rare], synth])
    ytr = np.concatenate([np.zeros(2000), np.ones(len(train_rare) + len(synth))])
    model = RandomForestClassifier(n_estimators=50, min_samples_leaf=3, n_jobs=-1,
                                   random_state=0).fit(Xtr, ytr)
    p1 = lambda Z: model.predict_proba(Z)[:, 1]
    t_rare = threshold(1 - p1(X[cal_rare])); t_com = threshold(p1(X[cal_com]))
    def label_sets(Z):
        p = p1(Z); return (p <= t_com), ((1 - p) <= t_rare)
    c1, r1 = label_sets(X[test_rare]); c2, r2 = label_sets(X[test_com])
    size = ((c1.astype(int) + r1).mean() + (c2.astype(int) + r2).mean()) / 2
    return r1.mean(), size

rng = np.random.default_rng(0)
rows = []
for m in M_LIST:
    n_rule = rule_n_cal(m)
    methods = {
        "A uniform 50/50, no rebalancing":  (int(round(0.5 * m)), 0),
        "B conventional 80/20, no rebal.":  (int(round(0.2 * m)), 0),
        "C rule, no rebalancing":           (n_rule, 0),
        "D rule + rebalancing":             (n_rule, 300),
        "E uniform 50/50 + rebalancing":    (int(round(0.5 * m)), 300),
    }
    for name, (n_cal, lam) in methods.items():
        res = [one_run(m, n_cal, lam, 1000 + r) for r in range(repeats)]
        cov = np.array([r[0] for r in res])
        rows.append(dict(m=m, method=name, n_cal=n_cal, rare_cov=cov.mean(),
                         reliable=(cov >= 1 - alpha - DELTA).mean(),
                         set_size=np.mean([r[1] for r in res])))
        print("done:", m, name, flush=True)
df = pd.DataFrame(rows)
df.to_csv("step07_results.csv", index=False)
print(df.round(3).to_string(index=False))