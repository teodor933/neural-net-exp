import numpy as np
from optimisers import *

class NeuralNetwork:
    def __init__(self, input_size, optimiser=None):
        self.input_size = input_size
        self.layers = []
        self.optimiser = optimiser if optimiser else SGD(learning_rate=1e-4)

    def add_layer(self, size, activation):
        prev_size = self.input_size if not self.layers else self.layers[-1][0].shape[0]
        weights = np.random.randn(size, prev_size) * 0.01
        biases = np.random.randn(size) * 0.01
        self.layers.append((weights, biases, activation))

    def forward(self, x):
        for weights, biases, activation in self.layers:
            x = activation.execute(np.dot(weights, x) + biases)
        return x

    def compute_loss(self, x, y_true, loss_fn):
        y_pred = self.forward(x)
        return loss_fn.execute(y_pred, y_true)

    def backpropagate(self, x, y_true, loss_fn):
        activations = [x]
        z_values = []
        a = x

        for weights, biases, activation in self.layers:
            z = np.dot(weights, a) + biases
            z_values.append(z)
            a = activation.execute(z)
            activations.append(a)

        loss_delta = loss_fn.derivative(activations[-1], y_true)

        for l in range(len(self.layers) - 1, -1, -1):
            weights, biases, activation = self.layers[l]
            z = z_values[l]

            act_derivative = activation.derivative(z)
            loss_delta *= act_derivative

            a_prev = activations[l]
            dW = np.outer(loss_delta, a_prev)
            db = loss_delta

            weights, biases = self.optimiser.update(weights, biases, dW, db)

            self.layers[l] = (weights, biases, activation)

            loss_delta = np.dot(weights.T, loss_delta)
