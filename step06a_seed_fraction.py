import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

alpha, m, rho, repeats = 0.1, 40, 0.5, 40
X, y = make_classification(n_samples=60000, n_features=20, n_informative=6,
                           weights=[0.98, 0.02], class_sep=0.8, random_state=0)
rare_all, common_all = np.where(y == 1)[0], np.where(y == 0)[0]

def threshold(scores):
    k = int(np.ceil((len(scores) + 1) * (1 - alpha)))
    return np.inf if k > len(scores) else np.sort(scores)[k - 1]

def smote(seeds, n_new, rng, k=5):
    k = min(k, len(seeds) - 1)                      # fewer seeds -> fewer neighbours
    d = np.linalg.norm(seeds[:, None] - seeds[None], axis=2)
    nbrs = np.argsort(d, axis=1)[:, 1:k + 1]
    i = rng.integers(len(seeds), size=n_new)
    j = nbrs[i, rng.integers(nbrs.shape[1], size=n_new)]
    return seeds[i] + rng.random((n_new, 1)) * (seeds[j] - seeds[i])

models = {"logistic": lambda: LogisticRegression(max_iter=1000),
          "forest":   lambda: RandomForestClassifier(n_estimators=50, min_samples_leaf=3,
                                                     n_jobs=-1, random_state=0)}
rng = np.random.default_rng(0)
lam = 300                                           # synthetic points generated
print("model     s     n_seeds  rare_coverage  avg_set_size (lower = better)")
for name, make in models.items():
    for s in [0.0, 0.25, 0.5, 1.0]:                 # s = 0 means no rebalancing at all
        cov, size = [], []
        for _ in range(repeats):
            rare = rng.permutation(rare_all); common = rng.permutation(common_all)
            test_rare, test_com = rare[:500], common[:1000]
            pool = rare[500:500 + m]; n_cal = int(round(rho * m))
            cal_rare, train_rare = pool[:n_cal], pool[n_cal:]
            cal_com = common[1000:2000]

            n_seeds = int(round(s * len(train_rare)))
            if n_seeds >= 3:
                seed_idx = rng.choice(train_rare, n_seeds, replace=False)  # seeds are train-only
                synth = smote(X[seed_idx], lam, rng)
            else:
                synth = np.empty((0, X.shape[1]))
            Xtr = np.vstack([X[common[2000:4000]], X[train_rare], synth])
            ytr = np.concatenate([np.zeros(2000), np.ones(len(train_rare) + len(synth))])
            model = make().fit(Xtr, ytr)
            p1 = lambda Z: model.predict_proba(Z)[:, 1]

            t_rare = threshold(1 - p1(X[cal_rare])); t_com = threshold(p1(X[cal_com]))
            def label_sets(Z):
                p = p1(Z); return (p <= t_com), ((1 - p) <= t_rare)
            c1, r1 = label_sets(X[test_rare]); c2, r2 = label_sets(X[test_com])
            cov.append(r1.mean())
            size.append(((c1.astype(int) + r1).mean() + (c2.astype(int) + r2).mean()) / 2)
        n_s = int(round(s * (m - n_cal)))
        print(f"{name:<9} {s:<5} {n_s:>7}  {np.mean(cov):>13.3f}  {np.mean(size):>12.2f}")