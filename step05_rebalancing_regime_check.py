import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

alpha, m, rho, repeats = 0.1, 100, 0.3, 30
X, y = make_classification(n_samples=60000, n_features=20, n_informative=6,
                           weights=[0.98, 0.02], class_sep=0.8, random_state=0)
rare_all, common_all = np.where(y == 1)[0], np.where(y == 0)[0]

def threshold(scores):
    k = int(np.ceil((len(scores) + 1) * (1 - alpha)))
    return np.inf if k > len(scores) else np.sort(scores)[k - 1]

def smote(seeds, n_new, rng, k=5):
    d = np.linalg.norm(seeds[:, None] - seeds[None], axis=2)
    nbrs = np.argsort(d, axis=1)[:, 1:k + 1]
    i = rng.integers(len(seeds), size=n_new)
    j = nbrs[i, rng.integers(nbrs.shape[1], size=n_new)]
    return seeds[i] + rng.random((n_new, 1)) * (seeds[j] - seeds[i])

models = {"logistic": lambda: LogisticRegression(max_iter=1000),
          "forest":   lambda: RandomForestClassifier(n_estimators=50, min_samples_leaf=3,
                                                     n_jobs=-1, random_state=0)}
rng = np.random.default_rng(0)
print("model     lambda  rare_coverage  avg_set_size (lower = better)")
for name, make in models.items():
    for lam in [0, 100, 300, 1000]:
        cov, size = [], []
        for _ in range(repeats):
            rare = rng.permutation(rare_all); common = rng.permutation(common_all)
            test_rare, test_com = rare[:500], common[:1000]
            pool = rare[500:500 + m]; n_cal = int(round(rho * m))
            cal_rare, train_rare = pool[:n_cal], pool[n_cal:]
            cal_com = common[1000:2000]

            synth = smote(X[train_rare], lam, rng) if lam else np.empty((0, X.shape[1]))
            Xtr = np.vstack([X[common[2000:4000]], X[train_rare], synth])
            ytr = np.concatenate([np.zeros(2000), np.ones(len(train_rare) + len(synth))])
            model = make().fit(Xtr, ytr)
            p1 = lambda Z: model.predict_proba(Z)[:, 1]

            t_rare = threshold(1 - p1(X[cal_rare]))     # one threshold per class,
            t_com = threshold(p1(X[cal_com]))           # from real calibration points only
            def label_sets(Z):
                p = p1(Z); return (p <= t_com), ((1 - p) <= t_rare)   # (common in set, rare in set)
            c1, r1 = label_sets(X[test_rare]); c2, r2 = label_sets(X[test_com])
            cov.append(r1.mean())                                   # rare coverage
            size.append(((c1.astype(int) + r1).mean() + (c2.astype(int) + r2).mean()) / 2)
        print(f"{name:<9} {lam:>6}  {np.mean(cov):>13.3f}  {np.mean(size):>12.2f}")