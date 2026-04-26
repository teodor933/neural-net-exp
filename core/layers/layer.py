from abc import ABC, abstractmethod

import numpy as np


class Layer(ABC):
    def __init__(self) -> None:
        self.built = False

        self.input_size: int | None = None
        self.output_size: int | None = None

        self.weights: np.ndarray | None = None
        self.biases: np.ndarray | None = None
        self.d_weights: np.ndarray | None = None
        self.d_biases: np.ndarray | None = None

        self.inputs: np.ndarray | None = None
        self.outputs: np.ndarray | None = None

    def build(self, input_size: int) -> None:
        self.input_size = input_size
        self.output_size = input_size
        self.built = True

    @abstractmethod
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def backward(self, d_loss: np.ndarray) -> np.ndarray:
        raise NotImplementedError
