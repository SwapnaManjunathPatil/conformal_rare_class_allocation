"""RQ3: how badly does coverage fail when the protocol leaks? (Step 4)  ->  results/leakage.csv

Four protocols, m = 40 real rare points, 16 of them for calibration:
  safe                      calibrate on real calibration points only
  synthetic_in_calibration  synthetic points are added to the calibration set
  calibration_in_training   the calibration points are also used to train the model
  calibration_as_seeds      the calibration points are also used as SMOTE seeds
"""
import sys
import numpy as np, pandas as pd
from conformal.core import threshold
from experiments.common import ALPHA, MODELS, RESULTS, synthetic_data, new_split, fit_predictor

MODES = ["safe", "synthetic_in_calibration", "calibration_in_training", "calibration_as_seeds"]


def run(quick=False):
    X, y = synthetic_data()
    m, n_cal = 40, 16
    reps = 4 if quick else 60
    lams = [0, 300] if quick else [0, 100, 300, 1000]
    rows = []
    for model in MODELS:
        for lam in lams:
            acc = {mode: [] for mode in MODES}
            for r in range(reps):
                sp, rng = new_split(y, 1000 + r, m)
                pool = sp["pool"]; cal, tr = pool[:n_cal], pool[n_cal:]

                pred, synth = fit_predictor(X, sp["train_com"], tr, lam, rng, model)
                ts = 1 - pred(sp["test_rare"]); s_cal = 1 - pred(cal)
                acc["safe"].append((ts <= threshold(s_cal, ALPHA)).mean())
                s_leak = np.concatenate([s_cal, 1 - pred.raw(synth)]) if lam else s_cal
                acc["synthetic_in_calibration"].append((ts <= threshold(s_leak, ALPHA)).mean())

                pr2, _ = fit_predictor(X, sp["train_com"], pool, lam, rng, model, seed_pts=tr)
                acc["calibration_in_training"].append(
                    ((1 - pr2(sp["test_rare"])) <= threshold(1 - pr2(cal), ALPHA)).mean())

                pr3, _ = fit_predictor(X, sp["train_com"], tr, lam, rng, model, seed_pts=pool)
                acc["calibration_as_seeds"].append(
                    ((1 - pr3(sp["test_rare"])) <= threshold(1 - pr3(cal), ALPHA)).mean())
            for mode in MODES:
                rows.append(dict(model=model, lam=lam, mode=mode,
                                 rare_cov=np.mean(acc[mode]), cov_sd=np.std(acc[mode])))
            print("leakage:", model, lam, flush=True)
    pd.DataFrame(rows).to_csv(RESULTS / "leakage.csv", index=False)


if __name__ == "__main__":
    run("--quick" in sys.argv)
