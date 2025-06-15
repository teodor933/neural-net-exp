import numpy as np
from typing import Optional, Tuple, List

from core.layers import Layer
from core.loss_functions import Loss


class NeuralNetwork:
    def __init__(self, input_size: Optional[int] = None) -> None:
        self.input_size = input_size
        self.layers: List[Layer] = []
        self.built = False

    def add_layer(self, layer: Layer) -> None:
        if not layer.built and not self.layers and self.input_size:
            layer.build(self.input_size)
        elif not layer.built and self.layers and self.layers[-1].built and hasattr(self.layers[-1], "output_size"):
            last_layer_output_size = self.layers[-1].output_size
            layer.build(last_layer_output_size)

        self.layers.append(layer)

    def build(self, input_size: int) -> None:
        if self.built:
            return

        current_size = input_size
        for layer in self.layers:
            if not layer.built:
                layer.build(current_size)
            if hasattr(layer, "output_size"):
                current_size = layer.output_size
        self.built = True

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        if not self.built: # check for any unbuilt layers due to lack of input_size specification
            start_size = self.input_size if self.input_size is not None else x.shape[0]
            if self.input_size is not None and self.input_size != x.shape[0]:
                raise ValueError(f"Input has {x.shape[0]}, but model expected {self.input_size}")
            self.build(start_size)

        activations = [x]

        for layer in self.layers:
            x = layer.forward(x)
            activations.append(x)
        return x, activations

    def predict(self, x: np.ndarray) -> np.ndarray:
        if not self.built:
            start_size = self.input_size if self.input_size is not None else x.shape[0]
            if self.input_size is not None and self.input_size != x.shape[0]:
                raise ValueError(f"Input has {x.shape[0]}, but model expected {self.input_size}")
            self.build(start_size)

        for layer in self.layers:
            x = layer.forward(x)
        return x

    def compute_loss(self, x: np.ndarray, y_true: np.ndarray, loss_fn: Loss) -> float:
        prediction = self.predict(x)
        return loss_fn.execute(prediction, y_true)

    def get_parameters(self):
        for layer in self.layers:
            yield layer.weights, layer.biases, layer.d_weights, layer.d_biases

    def backpropagation(self, d_loss: np.ndarray) -> np.ndarray:
        for layer in reversed(self.layers):
            d_loss, _, _ = layer.backward(d_loss) # derivatives
        return d_loss # just in case
