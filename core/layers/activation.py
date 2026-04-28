import numpy as np

from core.activation_functions.activation_function import ActivationFunction
from core.layers.layer import Layer


class Activation(Layer):
    def __init__(self, activation_fn: ActivationFunction) -> None:
        super().__init__()
        self.activation_fn = activation_fn

    def build(self, input_size: int) -> None:
        self.input_size = input_size
        self.output_size = input_size
        self.built = True

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        if not self.built:
            self.build(inputs.shape[1])

        self.inputs = inputs
        self.outputs = self.activation_fn.execute(inputs)
        return self.outputs

    def backward(self, d_loss: np.ndarray) -> np.ndarray:
        if self.inputs is None:
            raise RuntimeError("Cannot call backward() on Activation before forward().")
        return self.activation_fn.backward(self.inputs, d_loss) # dL/dz = dL/da * da/dz
