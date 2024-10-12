"""
Trains a Multi Layer Perceptron model, and then plots predictions. 
"""

import numpy as np
import argparse
from data_setup import setup
from utils import plot_predictions, save_model_weights
from model_builder import MultiLayerPerceptron, MeanSquaredError, GradientDescent

X_train, X_test, y_train, y_test = setup()

# Set up argument parsing
parser = argparse.ArgumentParser(description='Trains a PyTorch image classification model')

# Add arguments with type specifications
parser.add_argument('--model', type=str, default=None, help="Model to fit the data to.")
parser.add_argument('--lr', type=float, default=0.001, help="Learning rate for the optimizer")
parser.add_argument('--num_epochs', type=int, default=10, help="Number of epochs to train the model")
parser.add_argument('--hidden_units', type=int, default=128, help="Number of hidden units in the model")

args = parser.parse_args()

LEARNING_RATE = args.lr
NUM_EPOCHS = args.num_epochs
HIDDEN_UNITS = args.hidden_units

mlp = MultiLayerPerceptron(input_shape=1, hidden_units=HIDDEN_UNITS, output_shape=1)
loss_function = MeanSquaredError()
optimizer = GradientDescent(lr=LEARNING_RATE)

epochs = NUM_EPOCHS
np.random.seed(42)

for epoch in range(epochs):
    y_pred = mlp.forward(X_train)
    
    train_loss = loss_function.forward(y_train, y_pred)
    
    loss_gradients = loss_function.backward()
    mlp.backward(loss_gradients)
    
    optimizer.update_params(mlp.output_layer)
    optimizer.update_params(mlp.hidden_layer)
    optimizer.update_params(mlp.input_layer)
    
    if epoch % 100 == 0:
        print(f'Epoch {epoch} | Loss: {train_loss}')

test_pred = mlp.forward(X_test)

plot_predictions(X_train, y_train, 
                 X_test, y_test, 
                 predictions=test_pred)

save_model_weights(mlp)