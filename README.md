# Multi-Layer Perceptron (MLP) from Scratch
## Overview
This project implements a simple Multi-Layer Perceptron (MLP) from scratch in Python without the use of any external machine learning libraries (such as TensorFlow, PyTorch, or even NumPy). The MLP is designed to perform classification tasks, and includes forward propagation, backpropagation, and gradient descent for training the model.

## Features
Customizable neural network with a variable number of layers and neurons.
Implements ReLU activation for hidden layers.
Uses softmax activation for the output layer (for multi-class classification).
Trains the model using stochastic gradient descent (SGD).
Loss function: Cross-entropy for classification tasks.
## Architecture
The MLP architecture consists of:

Input layer: Receives the input data (features).
Hidden layers: One or more layers with ReLU activations.
Output layer: Produces the final predictions using softmax activation (for classification).
