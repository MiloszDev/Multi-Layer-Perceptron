import numpy as np

X = [[1, 2, 3, 2.5],
     [2.0, 5.0, -1.0, 2.0],
     [-1.5, 2.7, 3.3, -0.8]]

class LinearLayer:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        return np.dot(inputs, self.weights) + self.biases

class ReLU:
    def forward(self, inputs):
        return np.maximum(0, inputs)

linear_layer = LinearLayer(4, 5)
activation_function = ReLU()
print('tak')
print(linear_layer)
print(activation_function)

linear_layer = linear_layer.forward(X)

print(linear_layer)

linear_layer = activation_function.forward(linear_layer)

print(linear_layer)

class MultiLayerPerceptron:
    def __init__(self):
        pass
    