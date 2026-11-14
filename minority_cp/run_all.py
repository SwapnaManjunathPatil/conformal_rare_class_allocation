"""Run every experiment, then make every figure.

  python run_all.py --quick     a few minutes; tiny repeats, only to check that everything works
  python run_all.py             the full study (roughly 20-40 minutes on a laptop)
"""
import sys, time
from experiments import (exp_min_size, exp_regime_map, exp_leakage, exp_allocation_rule,
                         exp_cross_conformal, exp_real_datasets)
from analysis import make_figures

quick = "--quick" in sys.argv
for mod in (exp_min_size, exp_regime_map, exp_leakage, exp_allocation_rule,
            exp_cross_conformal, exp_real_datasets):
    t = time.time(); mod.run(quick); print(f"== {mod.__name__} finished in {time.time() - t:.0f}s\n")
make_figures.run()
