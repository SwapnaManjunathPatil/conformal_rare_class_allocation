# Conformal rare-class allocation

Joint allocation of **training**, **rebalancing**, and **conformal-calibration** budgets when the rare class has few labeled examples.

The question this repo answers: *who gets the minority samples?* Calibration needs real rare points for a valid threshold. Training and SMOTE also want those points. Spending them in the wrong place either breaks coverage or wastes set size.

The current study, advisor, and user-facing API live in [`minority_cp/`](minority_cp/).

## Quick start

```bash
cd minority_cp
pip install -r requirements.txt
python tests/test_core.py
python tests/test_advisor.py
python run_all.py --quick
```

`python run_all.py` is the full study (about 15–40 minutes). Runs are seeded, so they reproduce the CSVs in `minority_cp/results/` and the figures in `minority_cp/figures/`.

For real-data benchmarks (optional):

```bash
pip install imbalanced-learn
python -m experiments.exp_real_datasets mammography yeast_me2 us_crime
python -m analysis.make_figures
```

## Use it on your data

```bash
python advise.py --m 60 --alpha 0.1
python examples/use_on_your_data.py
```

```python
from allocation.advisor import recommend
from allocation.api import RareClassConformal

print(recommend(m=60, alpha=0.1))
cp = RareClassConformal(alpha=0.1).fit(X, y)  # y: 1 = rare class
sets = cp.predict_sets(X_new)                 # (n, 2) booleans: [common, rare]
```

Keep a final test set outside `fit` and check rare-class coverage on it. The advice is backed by the experiments for `alpha = 0.1`, cross-conformal at `m = 15–50`, and the calibration rule at `m = 30–120`.

## Leakage-safe protocol

Used throughout `minority_cp/`:

1. Split the real rare points into calibration and training first.
2. Generate synthetic points only from training points (never from calibration).
3. Calibrate on real points the model never trained on.

## Research questions

| Question | Code | Figure |
|---|---|---|
| RQ1 minimum calibration size | `minority_cp/experiments/exp_min_size.py` | `fig1_min_size.png` |
| RQ2 when rebalancing helps | `minority_cp/experiments/exp_regime_map.py` | `fig2_regime_map.png` |
| RQ3 leakage and the safe protocol | `minority_cp/experiments/exp_leakage.py` | `fig3_leakage.png` |
| RQ4 allocation rule vs baselines | `minority_cp/experiments/exp_allocation_rule.py` | `fig4_rule_vs_baselines.png` |
| RQ5 small-`m` reuse | `minority_cp/experiments/exp_cross_conformal.py` | `fig5_small_m.png` |
| Real datasets | `minority_cp/experiments/exp_real_datasets.py` | `fig6_real_datasets.png` |

Run one experiment with `python -m experiments.exp_leakage` (add `--quick` for a short version). Details: [`minority_cp/README.md`](minority_cp/README.md).

## Repository layout

```
minority_cp/     current code, advisor, experiments, results, figures
old_version/     earlier snapshot of the same study
step01_*.py …    notebook-style scripts used while building the study
```

Inside `minority_cp/`:

- `conformal/` — threshold, minimum size, over-coverage, cross-conformal
- `rebalance/` — SMOTE
- `allocation/` — calibration-size rule, `recommend()`, `RareClassConformal`
- `experiments/` — shared setup plus one script per question
- `analysis/` — figures
- `tests/` — sanity checks

## Known limits

One synthetic dataset plus two semi-natural real ones; forest and logistic models; 30–60 repeats. Cross-conformal has a weaker proven worst-case guarantee (about `1 - 2*alpha`) and is not proven here for the class-conditional case. The Xie et al. (2025) frequency-based splitting baseline and RC3P are not implemented.
