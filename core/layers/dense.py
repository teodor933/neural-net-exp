import numpy as np

from core.layers.layer import Layer
from core.weight_initializers.glorot_normal import GlorotNormal
from core.weight_initializers.weight_initializer import WeightInitializer


class Dense(Layer):
    def __init__(
            self,
            output_size: int,
            input_size: int | None = None,
            weight_initializer: WeightInitializer | None = None
    ) -> None:
        super().__init__()
        self.output_size = output_size
        self.weight_initializer = weight_initializer or GlorotNormal()

        if input_size is not None:
            self.build(input_size)

    def build(self, input_size: int) -> None:
        if self.built:
            return

        self.input_size = input_size
        self.weights = self.weight_initializer(input_size, self.output_size)
        self.biases = np.zeros((1, self.output_size))
        self.built = True

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        if not self.built:
            self.build(inputs.shape[1])

        self.inputs = inputs
        self.outputs = inputs @ self.weights + self.biases
        return self.outputs

    def backward(self, d_z: np.ndarray) -> np.ndarray:
        if self.inputs is None or self.weights is None or self.biases is None:
            raise RuntimeError("Cannot call backward() on Dense before forward() and build().")

        self.d_weights = self.inputs.T @ d_z  # dL/dW = dL/dz * dz/dW
        self.d_biases = np.sum(d_z, axis=0, keepdims=True)  # axis for rows instead
        d_inputs = d_z @ self.weights.T  # dL/dx = dL/dz * dz/dx
        return d_inputs
