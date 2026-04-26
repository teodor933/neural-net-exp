from abc import ABC, abstractmethod

import numpy as np

from core.parameters import Parameter


class Layer(ABC):
    def __init__(self) -> None:
        self.built = False

        self.input_size: int | None = None
        self.output_size: int | None = None

        self.inputs: np.ndarray | None = None
        self.outputs: np.ndarray | None = None

        self._parameters: list[Parameter] = []

    def build(self, input_size: int) -> None:
        self.input_size = input_size
        self.output_size = input_size
        self.built = True

    def parameters(self) -> list[Parameter]:
        return self._parameters

    def zero_gradient(self) -> None:
        for parameter in self.parameters():
            parameter.zero_gradient()

    @abstractmethod
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def backward(self, d_loss: np.ndarray) -> np.ndarray:
        raise NotImplementedError
