# Who gets the minority samples?
Joint allocation of training, rebalancing and conformal-calibration budgets under class imbalance.

## Run it
    pip install -r requirements.txt
    python tests/test_core.py          # sanity checks (seconds)
    python tests/test_advisor.py       # checks for the advisor and the user-facing class
    python run_all.py --quick          # smoke test, about 30 seconds
    python run_all.py                  # full study, about 15-40 minutes
Everything is seeded, so the full run reproduces the CSVs in `results/` and the figures in `figures/`.
Run single experiments with `python -m experiments.exp_leakage` (add `--quick` for a fast version).

## Use it on your own data
    python advise.py --m 60 --alpha 0.1                 # what should I do with 60 rare examples?
    python examples/use_on_your_data.py                 # full worked example (replace the two marked lines)

In code:

    from allocation.advisor import recommend
    from allocation.api import RareClassConformal
    print(recommend(m=60, alpha=0.1))                   # plan, warnings, expected coverage noise
    cp = RareClassConformal(alpha=0.1).fit(X, y)        # y: 1 = rare class; follows the plan, leakage-safe
    sets = cp.predict_sets(X_new)                       # (n, 2) booleans: [common in set, rare in set]

Keep a final test set outside `fit` and check rare-class coverage on it. The advice is backed by our
experiments only for alpha = 0.1, cross-conformal at m = 15-50 and the calibration rule at m = 30-120.

## Where each research question lives
| Question | Code | Output |
|---|---|---|
| RQ1 minimum calibration size | `experiments/exp_min_size.py`, `conformal/core.py` | `fig1_min_size.png` |
| RQ2 when does rebalancing help? | `experiments/exp_regime_map.py` | `fig2_regime_map.png` |
| RQ3 leakage and the safe protocol | `experiments/exp_leakage.py` | `fig3_leakage.png` |
| RQ4 allocation rule vs baselines | `allocation/rule.py`, `experiments/exp_allocation_rule.py` | `fig4_rule_vs_baselines.png` |
| RQ5 small-m reuse | `conformal/cross.py`, `experiments/exp_cross_conformal.py` | `fig5_small_m.png` |
| Real datasets | `experiments/exp_real_datasets.py` | `fig6_real_datasets.png` |

## Layout
    conformal/    threshold, minimum size, over-coverage, cross-conformal
    rebalance/    SMOTE
    allocation/   the calibration-size rule, advisor.py (recommend), api.py (RareClassConformal)
    experiments/  common.py (data, splits, models, split-CP and cross-CP) + one script per question
    analysis/     make_figures.py
    results/      CSV outputs        figures/   PNG outputs        tests/   sanity checks

## The leakage-safe protocol (used everywhere)
1. Split the real rare points into calibration and training first.
2. Synthetic points are generated only from training points (never from calibration points).
3. Calibration uses real points that the model never trained on.

## Real benchmark datasets
`pip install imbalanced-learn`, then for example
`python -m experiments.exp_real_datasets mammography yeast_me2 us_crime`
(downloads data on first use; writes `results/real_datasets.csv`, then rerun `python -m analysis.make_figures`).
The built-in `digits9` and `cancer_imb` are real scikit-learn data with the minority class thinned out.

## Known limits
One synthetic dataset plus two semi-natural real ones, forest and logistic models, 30-60 repeats.
Cross-conformal has only a weaker proven worst-case guarantee (about 1 - 2*alpha) and is not proven
here for the class-conditional case. The Xie et al. (2025) frequency-based splitting baseline and RC3P
are not implemented.
