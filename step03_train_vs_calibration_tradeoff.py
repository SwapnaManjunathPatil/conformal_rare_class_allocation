import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

alpha, m, repeats = 0.1, 400, 200          # m = number of real rare examples we own
X, y = make_classification(n_samples=60000, n_features=20, n_informative=6,
                           weights=[0.98, 0.02], class_sep=0.8, random_state=0)
rare_all, common_all = np.where(y == 1)[0], np.where(y == 0)[0]

def threshold(scores):
    k = int(np.ceil((len(scores) + 1) * (1 - alpha)))
    return np.inf if k > len(scores) else np.sort(scores)[k - 1]

rng = np.random.default_rng(0)
print("rho  n_cal  n_train  rare_coverage  coverage_spread  model_surprise")
for rho in [0.2, 0.4, 0.6, 0.8]:
    cov, surprise = [], []
    for _ in range(repeats):
        rare = rng.permutation(rare_all); common = rng.permutation(common_all)
        test_rare = rare[:500]
        pool = rare[500:500 + m]
        n_cal = int(round(rho * m))
        cal_rare, train_rare = pool[:n_cal], pool[n_cal:]

        train = np.concatenate([common[:2000], train_rare])
        model = LogisticRegression(max_iter=1000).fit(X[train], y[train])

        t = threshold(1 - model.predict_proba(X[cal_rare])[:, 1])   # score = 1 - P(rare)
        test_scores = 1 - model.predict_proba(X[test_rare])[:, 1]
        cov.append((test_scores <= t).mean())       # fraction of rare cases covered
        surprise.append(test_scores.mean())         # lower = better model
    print(f"{rho:.1f}  {n_cal:>5}  {m - n_cal:>7}  {np.mean(cov):>13.3f}  "
          f"{np.std(cov):>15.3f}  {np.mean(surprise):>14.3f}")