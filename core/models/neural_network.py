import numpy as np

from core.layers.layer import Layer


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

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        return self.forward(inputs)

    def get_parameters(self):
        for layer in self.layers:
            if layer.weights is not None and layer.biases is not None:
                yield layer.weights, layer.biases, layer.d_weights, layer.d_biases

    def summary(self) -> None:
        print("=== MODEL SUMMARY ===")
        for idx, layer in enumerate(self.layers):
            print(
                f"{idx:>2}: {layer.__class__.__name__:<15} "
                f"input={layer.input_size}, output={layer.output_size}"
            )

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        return self.forward(inputs)