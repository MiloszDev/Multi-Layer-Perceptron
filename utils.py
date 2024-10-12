"""
Contains various utility functions for plotting predictions and saving model weights to a file.
"""

import matplotlib.pyplot as plt
import pickle

def plot_predictions(train_data,
                     train_label,
                     test_data,
                     test_label,
                     predictions=None):
    """
    Plots training, test data and compare predictions. 
    """
    plt.figure(figsize=(10, 6))

    # Plot training data and testing data
    plt.scatter(train_data, train_label, c='b', s=4, label='Training Data')
    plt.scatter(test_data, test_label, c='g', s=4, label='Testing Data')

    if predictions is not None:
        plt.scatter(test_data, predictions, c='r', s=4, label='Predictions')
    plt.legend(prop={"size": 14})
    plt.show()

def save_model_weights(model, filename='model_weights.pkl'):
    model_weights = {
        'input_layer_weights': model.input_layer.weights,
        'input_layer_biases': model.input_layer.biases,
        'hidden_layer_weights': model.hidden_layer.weights,
        'hidden_layer_biases': model.hidden_layer.biases,
        'output_layer_weights': model.output_layer.weights,
        'output_layer_biases': model.output_layer.biases
    }

    with open(filename, 'wb') as file:
        pickle.dump(model_weights, file)

    print(f"Model weights saved to {filename}")

