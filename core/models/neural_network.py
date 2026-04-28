from typing import Iterator

import numpy as np

from core.layers.layer import Layer
from core.parameters import Parameter


class NeuralNetwork:
    def __init__(self, input_size: int | None = None) -> None:
        self.input_size = input_size
        self.layers: list[Layer] = []

    def add_layer(self, layer: Layer) -> None:
        if not self.layers:
            if self.input_size is not None and not layer.built:
                layer.build(self.input_size)
        else:
            previous_output_size = self.layers[-1].output_size
            if previous_output_size is not None and not layer.built:
                layer.build(previous_output_size)

        self.layers.append(layer)

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        x = inputs
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, d_loss: np.ndarray) -> np.ndarray:
        grad = d_loss
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def get_parameters(self) -> Iterator[Parameter]:
        for layer in self.layers:
            yield from layer.parameters()

    def zero_gradient(self) -> None:
        for parameter in self.get_parameters():
            parameter.zero_gradient()

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        previous_modes = [layer.training for layer in self.layers]

        self.eval()
        predictions = self.forward(inputs)

        # Switch back if needed to training mode
        for layer, was_training in zip(self.layers, previous_modes):
            if was_training:
                layer.train()
            else:
                layer.eval()

        return predictions

    def train(self) -> None:
        for layer in self.layers:
            layer.train()

    def eval(self) -> None:
        for layer in self.layers:
            layer.eval()

    def summary(self) -> None:
        print("=== MODEL SUMMARY ===")

        total_parameters = 0

        for idx, layer in enumerate(self.layers):
            trainable_parameters = sum(param.data.size for param in layer.parameters() if param.trainable)
            non_trainable_parameters = sum(param.data.size for param in layer.parameters() if not param.trainable)
            total_parameters += trainable_parameters + non_trainable_parameters

            print(
                f"{idx:>2}: {layer.__class__.__name__:<15} "
                f"input={layer.input_size}, output={layer.output_size}, "
                f"trainable_params={trainable_parameters}, non_trainable_params={non_trainable_parameters}, "
                f"layer_params={trainable_parameters + non_trainable_parameters}"
            )

        print(f"Total parameters: {total_parameters}")

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        return self.forward(inputs)