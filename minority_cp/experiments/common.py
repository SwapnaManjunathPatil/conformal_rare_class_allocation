"""Shared experiment helpers: data, splits, models, split-CP and cross-CP runs."""
from pathlib import Path
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from conformal.core import threshold
from conformal.cross import cross_conformal_inclusion
from rebalance.smote import smote

ROOT = Path(__file__).resolve().parents[1]
RESULTS, FIGURES = ROOT / "results", ROOT / "figures"
RESULTS.mkdir(exist_ok=True); FIGURES.mkdir(exist_ok=True)
ALPHA, LAM, K = 0.1, 300, 5            # miscoverage, synthetic points, cross-conformal folds

MODELS = {"logistic": lambda: LogisticRegression(max_iter=1000),
          "forest": lambda: RandomForestClassifier(n_estimators=50, min_samples_leaf=3,
                                                   n_jobs=-1, random_state=0)}


# ------------------------------------------------------------------ data
def synthetic_data():
    return make_classification(n_samples=60000, n_features=20, n_informative=6,
                               weights=[0.98, 0.02], class_sep=0.8, random_state=0)


# Real imbalanced datasets fetched with scikit-learn from OpenML (internet needed, no extra install)
OPENML = ("mammography", "ozone-level-8hr", "pc1", "creditcard")


def load_dataset(name):
    """(X, y) with y = 1 for the minority class.
    'digits9' and 'cancer_imb': real scikit-learn data with the minority thinned out (offline).
    'mammography', 'ozone-level-8hr', 'pc1', 'creditcard': downloaded from OpenML via scikit-learn.
    Anything else (e.g. 'yeast_me2', 'us_crime', 'ecoli'): imbalanced-learn (pip install imbalanced-learn)."""
    if name in ("digits9", "cancer_imb"):
        rng = np.random.default_rng(0)
        if name == "digits9":
            from sklearn.datasets import load_digits
            d = load_digits(); lab = (d.target == 9).astype(int); n_keep = 50
        else:
            from sklearn.datasets import load_breast_cancer
            d = load_breast_cancer(); lab = (d.target == 0).astype(int); n_keep = 40
        keep = np.concatenate([np.where(lab == 0)[0],
                               rng.choice(np.where(lab == 1)[0], n_keep, replace=False)])
        X, y = d.data[keep], lab[keep]
    elif name in OPENML:
        import sklearn.datasets as skd
        from sklearn.impute import SimpleImputer
        try:
            d = skd.fetch_openml(name, version=1, as_frame=False)
        except Exception:                      # some OpenML files trip the default reader; use the other one
            d = skd.fetch_openml(name, version=1, as_frame=False, parser="liac-arff")
        lab = np.asarray(d.target).astype(str)
        values, counts = np.unique(lab, return_counts=True)
        y = (lab == values[np.argmin(counts)]).astype(int)       # the rarer label becomes 1
        X = SimpleImputer(strategy="median").fit_transform(np.asarray(d.data, dtype=float))
    else:
        from imblearn.datasets import fetch_datasets
        d = fetch_datasets()[name]
        X, y = d.data, (d.target == 1).astype(int)
    return StandardScaler().fit_transform(X), y


def new_split(y, seed, m=None, n_test_rare=500, n_train_com=2000):
    """Random split. 'pool' = the budget of m real rare points we may use (all remaining if None)."""
    rng = np.random.default_rng(seed)
    rare = rng.permutation(np.where(y == 1)[0]); common = rng.permutation(np.where(y == 0)[0])
    nt = min(1000, len(common) // 4)
    sp = dict(test_rare=rare[:n_test_rare], test_com=common[:nt], cal_com=common[nt:2 * nt],
              train_com=common[2 * nt:2 * nt + n_train_com],
              pool=rare[n_test_rare:] if m is None else rare[n_test_rare:n_test_rare + m])
    sp["test"] = np.concatenate([sp["test_rare"], sp["test_com"]])
    return sp, rng


# ------------------------------------------------------------------ models
class Predictor:
    def __init__(self, mod, X):
        self.mod, self.X = mod, X

    def __call__(self, idx):                       # P(rare) for rows of X
        return self.mod.predict_proba(self.X[idx])[:, 1]

    def raw(self, Z):                              # P(rare) for new feature rows (e.g. synthetic)
        return self.mod.predict_proba(Z)[:, 1]


def fit_predictor(X, train_com, train_rare, lam, rng, model="forest", seed_pts=None):
    """Train on common + real rare + lam synthetic points (SMOTE seeds default to train_rare)."""
    seeds = train_rare if seed_pts is None else seed_pts
    synth = smote(X[seeds], lam, rng) if lam else np.empty((0, X.shape[1]))
    Xtr = np.vstack([X[train_com], X[train_rare], synth])
    ytr = np.concatenate([np.zeros(len(train_com)), np.ones(len(train_rare) + len(synth))])
    return Predictor(MODELS[model]().fit(Xtr, ytr), X), synth


def summarize(incl_rare, incl_com, nr):
    """Rare-class coverage and average set size, from label-inclusion flags (rare tests first)."""
    cov = incl_rare[:nr].mean()
    size = ((incl_com[:nr].astype(int) + incl_rare[:nr]).mean() +
            (incl_com[nr:].astype(int) + incl_rare[nr:]).mean()) / 2
    return cov, size


# ------------------------------------------------------------------ conformal runs
def split_cp(X, sp, n_cal, lam, rng, model, alpha=ALPHA):
    """Leakage-safe dedicated split: first n_cal pool points calibrate, the rest train.
    Returns {'mondrian': (cov, size), 'marginal': (cov, size)}."""
    pool, nr = sp["pool"], len(sp["test_rare"])
    pred, _ = fit_predictor(X, sp["train_com"], pool[n_cal:], lam, rng, model)
    p = pred(sp["test"])
    s_r, s_c = 1 - pred(pool[:n_cal]), pred(sp["cal_com"])
    t_r, t_c = threshold(s_r, alpha), threshold(s_c, alpha)
    q = threshold(np.concatenate([s_r, s_c]), alpha)
    return {"mondrian": summarize((1 - p) <= t_r, p <= t_c, nr),
            "marginal": summarize((1 - p) <= q, p <= q, nr)}


def cross_cp(X, sp, lam, rng, model, k=K, alpha=ALPHA):
    """Cross-conformal: each fold's model never sees that fold (nor seeds from it)."""
    pool, cal_com, test, nr = sp["pool"], sp["cal_com"], sp["test"], len(sp["test_rare"])
    f_r, f_c = np.arange(len(pool)) % k, np.arange(len(cal_com)) % k
    P = np.zeros((k, len(test))); s_r = np.zeros(len(pool)); s_c = np.zeros(len(cal_com))
    for j in range(k):
        pred, _ = fit_predictor(X, sp["train_com"], pool[f_r != j], lam, rng, model)
        P[j] = pred(test)
        s_r[f_r == j] = 1 - pred(pool[f_r == j]); s_c[f_c == j] = pred(cal_com[f_c == j])
    in_r, in_c = cross_conformal_inclusion(P, s_r, f_r, s_c, f_c, alpha)
    return summarize(in_r, in_c, nr)
