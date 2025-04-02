import numpy as np

from optimisers import *

class NeuralNetwork:
    def __init__(self, input_size, optimiser=None, batch_size=1):
        self.input_size = input_size
        self.layers = []
        self.optimiser = optimiser

        self.batch_size = batch_size

        self.activations = []
        self.z_values = []

    def add_layer(self, size, activation):
        prev_size = self.input_size if not self.layers else self.layers[-1][0].shape[0]
        weights = np.random.randn(size, prev_size) * 0.01
        biases = np.zeros((size, 1))
        self.layers.append((weights, biases, activation))

    def forward(self, x):
        if x.shape[0] != self.input_size:
            x = x.reshape(self.input_size, -1)

        self.activations = [x]
        self.z_values = []

        a = x
        for weights, biases, activation in self.layers:
            z = np.dot(weights, a) + biases
            self.z_values.append(z)
            a = activation.execute(z)
            self.activations.append(a)
        return a

    def compute_loss(self, x, y_true, loss_fn):
        y_pred = self.forward(x)
        return loss_fn.execute(y_pred, y_true)

    def backpropagate(self, x, y_true, loss_fn):

        _ = self.forward(x)

        delta_loss = loss_fn.derivative(self.activations[-1], y_true)

        for l in range(len(self.layers) - 1, -1, -1):
            weights, biases, activation = self.layers[l]
            z = self.z_values[l]

            delta_loss *= activation.derivative(z)

            a_prev = self.activations[l]
            dW = np.dot(delta_loss, a_prev.T) / x.shape[1] # norm by batch size
            db = np.mean(delta_loss, axis=1, keepdims=True) # axis for rows instead

            if self.optimiser:
                weights, biases = self.optimiser.update(weights, biases, dW, db)
                self.layers[l] = (weights, biases, activation)

            if l > 0:
                delta_loss = np.dot(weights.T, delta_loss)
