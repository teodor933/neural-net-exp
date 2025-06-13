import numpy as np
from typing import Optional, Tuple, List

from core.activations import Activation
from core.initialisers import Initialiser
from core.layers import Layer
from core.loss_functions import Loss


class NeuralNetwork:
    def __init__(self, input_size: int) -> None:
        self.input_size = input_size
        self.layers = []

    def add_layer(self,
                  size: int,
                  activation: Activation,
                  weight_initialiser: Optional[Initialiser] = None
                  ) -> None:
        input_size = self.input_size if not self.layers else self.layers[-1].weights.shape[0]
        self.layers.append(
            Layer(input_size, size, activation, weight_initialiser)
        )

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        if x.shape[0] != self.input_size:
            x = x.reshape(self.input_size, -1)

        activations = [x]

        for layer in self.layers:
            x = layer.forward(x)
            activations.append(x)
        return x, activations

    def predict(self, x: np.ndarray) -> np.ndarray:
        if x.shape[0] != self.input_size:
            x = x.reshape(self.input_size, -1)

        for layer in self.layers:
            x = layer.forward(x)
        return x

    def compute_loss(self, x: np.ndarray, y_true: np.ndarray, loss_fn: Loss) -> float:
        y_pred, _ = self.forward(x)
        return loss_fn.execute(y_pred, y_true)

    def get_parameters(self):
        for layer in self.layers:
            yield layer.weights, layer.biases, layer.d_weights, layer.d_biases

    def backpropagation(self, d_loss: np.ndarray) -> np.ndarray:
        for layer in reversed(self.layers):
            d_loss, _, _ = layer.backward(d_loss) # derivatives
        return d_loss # just in case
