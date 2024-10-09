import numpy as np

def setup():

    weight = 0.7
    bias = 0.3

    X = np.arange(0, 1, 0.02).reshape(-1, 1)
    y = weight * X + bias

    train_split = int(0.8 * len(X))

    X_train, X_test = X[:train_split], X[train_split:]
    y_train, y_test = y[:train_split], y[train_split:]
    
    return X_train, X_test, y_train, y_test