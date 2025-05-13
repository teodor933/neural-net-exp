import numpy as np

from core.layers import Layer

class NeuralNetwork:
    def __init__(self, input_size):
        self.input_size = input_size
        self.layers = []

    def add_layer(self, size, activation, weight_initialiser=None):
        input_size = self.input_size if not self.layers else self.layers[-1].weights.shape[0]
        self.layers.append(
            Layer(input_size, size, activation, weight_initialiser)
        )

    def forward(self, x):
        if x.shape[0] != self.input_size:
            x = x.reshape(self.input_size, -1)

        activations = [x]

        for layer in self.layers:
            x = layer.forward(x)
            activations.append(x)
        return x, activations

    def predict(self, x):
        if x.shape[0] != self.input_size:
            x = x.reshape(self.input_size, -1)

        for layer in self.layers:
            x = layer.forward(x)
        return x

    def compute_loss(self, x, y_true, loss_fn):
        y_pred, _ = self.forward(x)
        return loss_fn.execute(y_pred, y_true)

    def backpropagate(self, d_loss, optimiser):
        for layer in reversed(self.layers):
            d_loss, _, _ = layer.backward(d_loss) # derivatives
            layer.update(optimiser) # alter weights according to derivatives
        return d_loss # just in case
