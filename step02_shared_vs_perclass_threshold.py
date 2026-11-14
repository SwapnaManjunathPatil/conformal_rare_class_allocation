import numpy as np
rng = np.random.default_rng(0)
alpha = 0.1

def threshold(scores):
    k = int(np.ceil((len(scores) + 1) * (1 - alpha)))
    return np.sort(scores)[k - 1]

# scores: how "surprised" the model is. Common class: low. Rare class: higher.
common_cal = rng.beta(2, 8, 1000)
rare_cal   = rng.beta(5, 4, 20)
common_test = rng.beta(2, 8, 100000)
rare_test   = rng.beta(5, 4, 100000)

t_shared = threshold(np.concatenate([common_cal, rare_cal]))
t_common, t_rare = threshold(common_cal), threshold(rare_cal)

print("ONE shared threshold:")
print("  common coverage:", (common_test <= t_shared).mean())
print("  rare coverage:  ", (rare_test <= t_shared).mean())
print("SEPARATE thresholds:")
print("  common coverage:", (common_test <= t_common).mean())
print("  rare coverage:  ", (rare_test <= t_rare).mean())