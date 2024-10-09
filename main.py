import numpy as np
from data_setup import setup

X_train, X_test, y_train, y_test = setup()

class LinearLayer:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        return np.dot(inputs, self.weights) + self.biases

class ReLU:
    def forward(self, inputs):
        return np.maximum(0, inputs)

class MeanSquaredError:
    def __init__(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred

        self.calculate_mae()

    def calculate_mae(self):
        return np.square(self.y_true - self.y_pred).mean()
    
    def backward(self):
        pass

epochs = 3

for i in range(epochs):
    y_pred = LinearLayer(1, 1).forward(X_train)

    train_loss = MeanSquaredError(y_train, y_pred)
    print(train_loss.calculate_mae())

    