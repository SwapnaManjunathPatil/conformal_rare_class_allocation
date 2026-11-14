"""Minimal SMOTE (Step 4): interpolate between a seed and one of its k nearest neighbours."""
import numpy as np


def smote(seeds, n_new, rng, k=5):
    if len(seeds) < 3 or n_new == 0:
        return np.empty((0, seeds.shape[1]))
    k = min(k, len(seeds) - 1)
    d = np.linalg.norm(seeds[:, None] - seeds[None], axis=2)
    nbrs = np.argsort(d, axis=1)[:, 1:k + 1]
    i = rng.integers(len(seeds), size=n_new)
    j = nbrs[i, rng.integers(nbrs.shape[1], size=n_new)]
    return seeds[i] + rng.random((n_new, 1)) * (seeds[j] - seeds[i])
