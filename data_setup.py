import numpy as np

weight = 0.7
bias = 0.3

X = np.arange(0, 1, 0.2).reshape(-1, 1)
y = weight * X + bias

len(X)