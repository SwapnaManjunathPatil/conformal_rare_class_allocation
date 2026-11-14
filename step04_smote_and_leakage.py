import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

alpha, m, rho, repeats = 0.1, 40, 0.4, 200
X, y = make_classification(n_samples=60000, n_features=20, n_informative=6,
                           weights=[0.98, 0.02], class_sep=0.8, random_state=0)
rare_all, common_all = np.where(y == 1)[0], np.where(y == 0)[0]

def threshold(scores):
    k = int(np.ceil((len(scores) + 1) * (1 - alpha)))
    return np.inf if k > len(scores) else np.sort(scores)[k - 1]

def smote(seeds, n_new, rng, k=5):
    """Make n_new synthetic points by interpolating between a seed and a near neighbour."""
    d = np.linalg.norm(seeds[:, None] - seeds[None], axis=2)
    nbrs = np.argsort(d, axis=1)[:, 1:k + 1]
    out = []
    for _ in range(n_new):
        i = rng.integers(len(seeds)); j = rng.choice(nbrs[i])
        out.append(seeds[i] + rng.random() * (seeds[j] - seeds[i]))
    return np.array(out)

rng = np.random.default_rng(0)
print("lambda  safe_coverage  leaky_coverage")
for lam in [0, 100, 300, 1000]:
    safe, leaky = [], []
    for _ in range(repeats):
        rare = rng.permutation(rare_all); common = rng.permutation(common_all)
        test_rare = rare[:500]; pool = rare[500:500 + m]
        n_cal = int(round(rho * m)); cal_rare, train_rare = pool[:n_cal], pool[n_cal:]

        synth = smote(X[train_rare], lam, rng) if lam > 0 else np.empty((0, X.shape[1]))
        Xtr = np.vstack([X[common[:2000]], X[train_rare], synth])
        ytr = np.concatenate([np.zeros(2000), np.ones(len(train_rare) + len(synth))])
        model = LogisticRegression(max_iter=1000).fit(Xtr, ytr)

        score = lambda Z: 1 - model.predict_proba(Z)[:, 1]
        test_s = score(X[test_rare])
        t_safe = threshold(score(X[cal_rare]))                          # real points only
        t_leak = threshold(np.concatenate([score(X[cal_rare]),
                                           score(synth) if lam else []]))  # synthetic leaked in
        safe.append((test_s <= t_safe).mean()); leaky.append((test_s <= t_leak).mean())
    print(f"{lam:>6}  {np.mean(safe):>13.3f}  {np.mean(leaky):>14.3f}")