import numpy as np

from core.layers.layer import Layer
from core.parameters import Parameter
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

        self.weights: Parameter | None = None
        self.biases: Parameter | None = None

        if input_size is not None:
            self.build(input_size)

    def build(self, input_size: int) -> None:
        if self.built:
            return

        self.input_size = input_size

        weights = self.weight_initializer(input_size, self.output_size)
        biases = np.zeros((1, self.output_size), dtype=np.float32)
        self.weights = Parameter(weights, name="weights")
        self.biases = Parameter(biases, name="biases")

        self._parameters = [self.weights, self.biases]

        self.built = True

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        if not self.built:
            self.build(inputs.shape[1])

        if self.weights is None or self.biases is None:
            raise RuntimeError("Dense layer was not built correctly.")

        self.inputs = inputs
        self.outputs = inputs @ self.weights.data + self.biases.data
        return self.outputs

    def backward(self, d_z: np.ndarray) -> np.ndarray:
        if self.inputs is None:
            raise RuntimeError("Cannot call backward() on Dense before forward().")

        if self.weights is None or self.biases is None:
            raise RuntimeError("Dense layer was not built correctly.")

        self.weights.gradient = self.inputs.T @ d_z  # dL/dW = dL/dz * dz/dW
        self.biases.gradient = np.sum(d_z, axis=0, keepdims=True)  # axis for rows instead
        d_inputs = d_z @ self.weights.data.T  # dL/dx = dL/dz * dz/dx
        return d_inputs
