"""Command-line advisor.   python advise.py --m 60 --alpha 0.1"""
import argparse
from allocation.advisor import recommend

ap = argparse.ArgumentParser(description="How should I use my rare-class examples?")
ap.add_argument("--m", type=int, required=True, help="number of real rare examples available")
ap.add_argument("--alpha", type=float, default=0.1, help="miscoverage, 0.1 = 90%% coverage")
a = ap.parse_args()
print(recommend(a.m, a.alpha))
