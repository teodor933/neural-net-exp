import numpy as np
from abc import ABC, abstractmethod

from core.activations import Activation
from core.initialisers import Initialiser

from typing import Tuple, Optional

class Layer(ABC):
    def __init__(self):
        self.weights = None
        self.biases = None
        self.d_weights = None
        self.d_biases = None
        self.built = False

        self.output_size = None

    def build(self, input_size: int):
        self.built = True

    @abstractmethod
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def backward(self, d_loss: np.ndarray) -> np.ndarray:
        raise NotImplementedError

class Dense(Layer):
    def __init__(self,
                 output_size: int,
                 activation: Activation,
                 weight_initialiser: Initialiser = None,
                 input_size: Optional[int] = None
                 ) -> None:
        super().__init__()
        self.output_size = output_size
        self.activation = activation
        self.weight_initialiser = weight_initialiser or getattr(activation, "weight_initialiser")

        self.inputs: Optional[np.ndarray] = None
        self.z: Optional[np.ndarray] = None
        self.outputs: Optional[np.ndarray] = None

        self.d_weights: Optional[np.ndarray] = None
        self.d_biases: Optional[np.ndarray] = None

        if input_size is not None:
            self.build(input_size)

    def build(self, input_size: int) -> None:
        if self.built:
            return

        self.weights = self.weight_initialiser(input_size, self.output_size)
        self.biases = np.zeros((self.output_size, 1))
        self.built = True

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        if not self.built:
            raise RuntimeError("Layer has not been built yet. The method build() should be called before using the layer.")

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
