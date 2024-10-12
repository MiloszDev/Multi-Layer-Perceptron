import numpy as np

class LinearLayer:
    def __init__(self, n_inputs, n_neurons) -> None:
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases
    
    def backward(self, derative_values):
        self.derative_weights = np.dot(np.array(self.inputs).T, derative_values)
        self.derative_biases = np.sum(derative_values, axis=0, keepdims=True)

        self.derative_inputs = np.dot(derative_values, np.array(self.weights).T)
class ReLU:
    def forward(self, inputs):
        self.inputs = inputs
        
        self.output= np.maximum(0, inputs)
    def backward(self, derative_values):
        self.derative_inputs = derative_values.copy()
        self.derative_inputs[self.inputs <= 0] = 0
class MeanSquaredError:
    def __init__(self):
        pass
    def forward(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred
        
        return np.square(self.y_true - self.y_pred).mean()
    
    def backward(self):
        return 2 * (self.y_pred - self.y_true) / len(self.y_true)

class GradientDescent:
    def __init__(self, lr=0.01):
        self.learning_rate = lr

    def update_params(self, layer):
        layer.weights -= self.learning_rate * layer.derative_weights
        layer.biases -= self.learning_rate * layer.derative_biases

class MultiLayerPerceptron:
    def __init__(self, input_shape, hidden_units, output_shape):
        self.input_layer = LinearLayer(input_shape, hidden_units)
        self.hidden_layer = LinearLayer(hidden_units, hidden_units)
        self.output_layer = LinearLayer(hidden_units, output_shape)

        self.relu = ReLU()
    def forward(self, inputs):
        self.input_layer.forward(inputs)
        self.relu.forward(self.input_layer.output)

        self.hidden_layer.forward(self.relu.output)
        self.relu.forward(self.hidden_layer.output)

        self.hidden_layer.forward(self.relu.output)
        self.relu.forward(self.hidden_layer.output)
        
        self.output_layer.forward(self.relu.output)

        return self.output_layer.output
    
    def backward(self, loss_gradient):
        self.output_layer.backward(loss_gradient)
        self.relu.backward(self.output_layer.derative_inputs)

        self.hidden_layer.backward(self.relu.derative_inputs)
        self.relu.backward(self.hidden_layer.derative_inputs)

        self.hidden_layer.backward(self.relu.derative_inputs)
        self.relu.backward(self.hidden_layer.derative_inputs)

        self.input_layer.backward(self.relu.derative_inputs)