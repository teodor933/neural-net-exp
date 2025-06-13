import numpy as np

from core.activations import Activation
from core.initialisers import Initialiser

from typing import Tuple, Optional


class Layer:
    def __init__(self,
                 input_size: int,
                 output_size: int,
                 activation: Activation,
                 weight_initialiser: Initialiser = None
                 ) -> None:
        self.weight_initialiser = weight_initialiser or getattr(activation, "weight_initialiser")

        self.weights = self.weight_initialiser(input_size, output_size)
        self.biases = np.zeros((output_size, 1))
        self.activation = activation

        self.inputs: Optional[np.ndarray] = None
        self.z: Optional[np.ndarray] = None
        self.outputs: Optional[np.ndarray] = None
        self.d_weights: Optional[np.ndarray] = None
        self.d_biases: Optional[np.ndarray] = None

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.inputs = inputs
        self.z = np.dot(self.weights, self.inputs) + self.biases
        self.outputs = self.activation.execute(self.z)
        return self.outputs

    def backward(self, d_loss: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        d_z = d_loss * self.activation.derivative(self.z) # dL/dz = dL/da * da/dz
        self.d_weights = np.dot(d_z, self.inputs.T) # dL/dW = dL/dz * dz/dW
        self.d_biases = np.sum(d_z, axis=1, keepdims=True) # axis for rows instead
        d_inputs = np.dot(self.weights.T, d_z) # dL/dx = dL/dz * dz/dx
        return d_inputs, self.d_weights, self.d_biases
