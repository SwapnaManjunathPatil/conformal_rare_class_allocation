import numpy as np
rng = np.random.default_rng(2)
alpha = 0.1                                   # we want 90% coverage
cal = rng.random(20)                        # 20 calibration scores
k = int(np.ceil((len(cal) + 1) * (1 - alpha)))
threshold = np.sort(cal)[k - 1]               # the k-th smallest score
test = rng.random(100000)                     # new scores from the same source
print("threshold:", threshold, "coverage:", (test <= threshold).mean())