"""One-call advice: given m rare examples and a target level, what should I do?

    from allocation.advisor import recommend
    print(recommend(m=60, alpha=0.1))

The advice follows the experiments in this project. Its evidence is limited to alpha = 0.1,
cross-conformal for m = 15-50, and the calibration-size rule for m = 30-120 (one synthetic
dataset and two semi-natural real ones), so the plan lists warnings where you leave that range.
"""
from dataclasses import dataclass, field
import numpy as np
from scipy.stats import beta
from conformal.core import min_calibration_size, slack
from allocation.rule import rule_n_cal

CROSS_MAX_M = 50          # largest m at which cross-conformal was tested
CROSS_MIN_M = 15          # smallest m at which cross-conformal was tested
FOLDS = 5


@dataclass
class Plan:
    m: int
    alpha: float
    strategy: str                    # "dedicated_split", "cross_conformal" or "infeasible"
    n_cal: int = None                # rare points held out for calibration (split only)
    n_train: int = None              # rare points used for training (per model for cross-conformal)
    level_used: float = None         # quantile level actually used by the split (>= 1 - alpha)
    coverage_sd: float = None        # expected run-to-run spread of rare-class coverage
    n_synthetic: int = 0
    warnings: list = field(default_factory=list)
    notes: list = field(default_factory=list)

    @property
    def feasible(self):
        return self.strategy != "infeasible"

    def __str__(self):
        lines = [f"Plan for m = {self.m} rare points, target coverage {1 - self.alpha:.0%}",
                 f"  Strategy: {self.strategy.replace('_', ' ')}"]
        if self.strategy == "dedicated_split":
            lines += [f"  Hold out {self.n_cal} rare points for calibration, train on {self.n_train}.",
                      f"  The threshold really uses level {self.level_used:.3f}; "
                      f"expect rare-class coverage to vary by about +/-{self.coverage_sd:.3f} between runs."]
        elif self.strategy == "cross_conformal":
            lines.append(f"  Reuse every rare point: {FOLDS} folds, each model trains on about "
                         f"{self.n_train} rare points and calibrates on the rest.")
        if self.n_synthetic:
            lines.append(f"  Rebalancing: optional, about {self.n_synthetic} synthetic points made only "
                         f"from training points.")
        for n in self.notes:
            lines.append(f"  Note: {n}")
        for w in self.warnings:
            lines.append(f"  WARNING: {w}")
        return "\n".join(lines)


def _split_stats(n, alpha):
    l = int(np.floor((n + 1) * alpha))
    sd = float(np.sqrt((n + 1 - l) * l / ((n + 1) ** 2 * (n + 2)))) if l > 0 else float("nan")
    return float(np.ceil((n + 1) * (1 - alpha)) / (n + 1)), sd


def recommend(m, alpha=0.1, delta=0.05, eta=0.10, n_synthetic=300):
    """Return a Plan for m real rare examples (after setting aside any final test set)."""
    n_min = min_calibration_size(alpha)
    if m < n_min:
        a_min = float(np.ceil(100 / (m + 1))) / 100
        return Plan(m, alpha, "infeasible", warnings=[
            f"With {m} rare points no useful class-conditional guarantee exists at level {1 - alpha:.0%}: "
            f"you need at least {n_min} for calibration alone, and the model also needs rare examples to "
            f"train on. Collect more rare data, or accept a lower level (alpha of about {a_min} or more)."])

    plan = Plan(m, alpha, "cross_conformal", n_synthetic=n_synthetic)
    n_rule = rule_n_cal(m, alpha, delta, eta)
    use_split = m > CROSS_MAX_M and n_rule is not None and m - n_rule >= 5
    if use_split:
        plan.strategy, plan.n_cal, plan.n_train = "dedicated_split", n_rule, m - n_rule
        plan.level_used, plan.coverage_sd = _split_stats(n_rule, alpha)
        plan.notes.append("The guarantee is the standard one: rare-class coverage of at least "
                          f"{1 - alpha:.0%} on average, if calibration points are exchangeable with new cases.")
        if m > 120:
            plan.notes.append("With this many points a larger calibration set (for example half of them) "
                              "gave more reliable coverage in one test; consider it if worst-case coverage matters.")
    else:
        plan.n_train = int(round(m * (FOLDS - 1) / FOLDS))
        plan.notes.append("Cross-conformal gave steadier coverage than a dedicated split at small m in our "
                          "tests, but its proven worst-case guarantee is weaker (about 1 - 2*alpha) and has "
                          "not been proven for the class-conditional case.")
        if m < CROSS_MIN_M:
            plan.warnings.append(f"m = {m} is below the range we tested (15-50); treat results as a rough guide.")
    plan.notes.append("Use one threshold per class (Mondrian). A single shared threshold covered 0% and 39% "
                      "of the rare class on our two real datasets.")
    plan.notes.append("Never put synthetic points, or points the model trained on or used as seeds, "
                      "into calibration. In our forest tests this dropped coverage from 0.95 to near zero.")
    plan.notes.append("Rebalancing helped a flexible model (forest) most at small m and changed little for a "
                      "linear model; check it on your data.")
    if abs(alpha - 0.1) > 1e-9:
        plan.warnings.append("Our experiments used alpha = 0.1; the theory applies to other levels but "
                             "the numbers here were not tested.")
    return plan


# ---------------------------------------------------------------- leakage guards
def leakage_safe_split(rare_idx, n_cal, rng):
    """Shuffle the rare indices and return (train_idx, cal_idx) with no overlap."""
    rare_idx = rng.permutation(np.asarray(rare_idx))
    return rare_idx[n_cal:], rare_idx[:n_cal]


def check_no_leakage(train_idx, cal_idx, seed_idx=None):
    """Raise if any calibration point is also a training point or a synthetic-data seed."""
    cal = set(np.asarray(cal_idx).tolist())
    if cal & set(np.asarray(train_idx).tolist()):
        raise ValueError("Leakage: some calibration points are also in the training set.")
    if seed_idx is not None and cal & set(np.asarray(seed_idx).tolist()):
        raise ValueError("Leakage: some calibration points are used as seeds for synthetic data.")
    return True
