"""Apply the recommended plan to your own data.

    from allocation.api import RareClassConformal
    cp = RareClassConformal(alpha=0.1).fit(X_train, y_train)     # y: 1 = rare class, 0 = everything else
    sets = cp.predict_sets(X_new)       # boolean array (n, 2): column 0 = "common" in set, 1 = "rare" in set
    print(cp.plan_)                     # what it decided and why
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from conformal.core import threshold, min_calibration_size
from conformal.cross import cross_conformal_inclusion
from rebalance.smote import smote
from allocation.advisor import recommend, leakage_safe_split, check_no_leakage, FOLDS

_MODELS = {"forest": lambda: RandomForestClassifier(n_estimators=50, min_samples_leaf=3,
                                                    n_jobs=-1, random_state=0),
           "logistic": lambda: LogisticRegression(max_iter=1000)}


class RareClassConformal:
    """Binary rare-class conformal prediction with one threshold per class and a leakage-safe split.

    n_synthetic = 0 turns rebalancing off. strategy = None follows the advisor;
    "split" or "cross" forces one. Common-class calibration uses 25% of the common points (at most 1000).
    """
    def __init__(self, alpha=0.1, model="forest", n_synthetic=300, strategy=None, seed=0):
        self.alpha, self.model, self.n_synthetic, self.strategy, self.seed = alpha, model, n_synthetic, strategy, seed

    def _fit_one(self, X, train_com, train_rare, rng):
        synth = smote(X[train_rare], self.n_synthetic, rng) if self.n_synthetic else np.empty((0, X.shape[1]))
        Xtr = np.vstack([X[train_com], X[train_rare], synth])
        ytr = np.concatenate([np.zeros(len(train_com)), np.ones(len(train_rare) + len(synth))])
        mod = _MODELS[self.model]().fit(Xtr, ytr)
        return lambda Z: mod.predict_proba(Z)[:, 1]

    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y).astype(int)
        rng = np.random.default_rng(self.seed)
        rare, common = rng.permutation(np.where(y == 1)[0]), rng.permutation(np.where(y == 0)[0])
        m = len(rare)
        plan = recommend(m, self.alpha, n_synthetic=self.n_synthetic)
        if not plan.feasible:
            raise ValueError(str(plan))
        n_c = min(1000, len(common) // 4)
        if n_c < min_calibration_size(self.alpha):
            raise ValueError("Too few common-class points to calibrate the common class.")
        cal_com, train_com = common[:n_c], common[n_c:n_c + 4000]
        strategy = {"split": "dedicated_split", "cross": "cross_conformal"}.get(self.strategy, plan.strategy)
        self.plan_, self.strategy_ = plan, strategy
        if strategy == "dedicated_split":
            n_cal = plan.n_cal or max(min_calibration_size(self.alpha), m // 2)
            tr_r, cal_r = leakage_safe_split(rare, n_cal, rng)
            check_no_leakage(tr_r, cal_r, seed_idx=tr_r)
            p = self._fit_one(X, train_com, tr_r, rng)
            self.t_ = (threshold(p(X[cal_com]), self.alpha), threshold(1 - p(X[cal_r]), self.alpha))
            self.p_ = p
        else:
            f_r, f_c = np.arange(m) % FOLDS, np.arange(len(cal_com)) % FOLDS
            self.models_ = []; s_r = np.zeros(m); s_c = np.zeros(len(cal_com))
            for k in range(FOLDS):
                check_no_leakage(rare[f_r != k], rare[f_r == k], seed_idx=rare[f_r != k])
                p = self._fit_one(X, train_com, rare[f_r != k], rng)
                self.models_.append(p)
                s_r[f_r == k] = 1 - p(X[rare[f_r == k]]); s_c[f_c == k] = p(X[cal_com[f_c == k]])
            self.cross_ = (s_r, f_r, s_c, f_c)
        return self

    def predict_sets(self, X):
        X = np.asarray(X, float)
        if self.strategy_ == "dedicated_split":
            p = self.p_(X)
            return np.stack([p <= self.t_[0], (1 - p) <= self.t_[1]], axis=1)
        P = np.stack([p(X) for p in self.models_])
        s_r, f_r, s_c, f_c = self.cross_
        in_r, in_c = cross_conformal_inclusion(P, s_r, f_r, s_c, f_c, self.alpha)
        return np.stack([in_c, in_r], axis=1)
