import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

alpha, rho, repeats = 0.1, 0.5, 30
M_LIST, LAM_LIST = [20, 40, 100, 200], [0, 100, 300, 1000]
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

models = {"logistic": lambda: LogisticRegression(max_iter=1000),
          "forest":   lambda: RandomForestClassifier(n_estimators=50, min_samples_leaf=3,
                                                     n_jobs=-1, random_state=0)}

def one_run(make, m, lam, rng):
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
    t_rare = threshold(1 - p1(X[cal_rare])); t_com = threshold(p1(X[cal_com]))
    def label_sets(Z):
        p = p1(Z); return (p <= t_com), ((1 - p) <= t_rare)
    c1, r1 = label_sets(X[test_rare]); c2, r2 = label_sets(X[test_com])
    size = ((c1.astype(int) + r1).mean() + (c2.astype(int) + r2).mean()) / 2
    return r1.mean(), size

rng = np.random.default_rng(0)
rows = []
for name, make in models.items():
    for m in M_LIST:
        for lam in LAM_LIST:
            res = [one_run(make, m, lam, rng) for _ in range(repeats)]
            rows.append(dict(model=name, m=m, lam=lam,
                             rare_cov=np.mean([r[0] for r in res]),
                             cov_sd=np.std([r[0] for r in res]),
                             set_size=np.mean([r[1] for r in res])))
df = pd.DataFrame(rows)
df.to_csv("step06b_results.csv", index=False)

# Regime map: change in set size vs no rebalancing (lambda = 0). Negative = rebalancing helps.
fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
for ax, name in zip(axes, models):
    d = df[df.model == name].pivot(index="m", columns="lam", values="set_size")
    delta = d.sub(d[0], axis=0).drop(columns=0)
    im = ax.imshow(delta.values, cmap="RdBu_r", vmin=-0.25, vmax=0.25, aspect="auto")
    ax.set_xticks(range(delta.shape[1])); ax.set_xticklabels(delta.columns)
    ax.set_yticks(range(delta.shape[0])); ax.set_yticklabels(delta.index)
    ax.set_xlabel("synthetic points (lambda)"); ax.set_ylabel("real rare points (m)")
    ax.set_title(f"{name}: change in set size")
    for (i, j), v in np.ndenumerate(delta.values):
        ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=10)
fig.colorbar(im, ax=axes, label="blue = smaller sets (helps), red = larger (hurts)")
plt.savefig("step06b_regime_map.png", dpi=150, bbox_inches="tight")
print(df.round(3).to_string(index=False))
print("\nworst rare coverage over grid:", df.rare_cov.min().round(3))