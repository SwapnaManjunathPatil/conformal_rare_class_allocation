"""Cross-conformal reuse for the rare class (Step 8)."""
import numpy as np


def cross_conformal_inclusion(P, s_rare, f_rare, s_com, f_com, alpha):
    """Decide, for every test point, whether each label is in its prediction set.

    P[k, j]   : P(rare | x_j) from the model of fold k (never trained on fold k)
    s_rare[i] : score 1 - P(rare | x_i) of rare calibration point i, from its own fold's model
    s_com[i]  : score P(rare | x_i) of common calibration point i, from its own fold's model
    f_*       : fold index of each calibration point
    A label is kept when its p-value (1 + #calibration scores >= test score) / (n + 1) > alpha.
    """
    in_rare = (1 + (s_rare[:, None] >= (1 - P[f_rare])).sum(0)) / (len(s_rare) + 1) > alpha
    in_com = (1 + (s_com[:, None] >= P[f_com]).sum(0)) / (len(s_com) + 1) > alpha
    return in_rare, in_com
