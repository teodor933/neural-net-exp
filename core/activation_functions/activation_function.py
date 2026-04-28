from abc import ABC, abstractmethod

import numpy as np


class ActivationFunction(ABC):
    @abstractmethod
    def execute(self, inputs: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def derivative(self, inputs: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def backward(self, inputs: np.ndarray, d_outputs: np.ndarray) -> np.ndarray:
        return d_outputs * self.derivative(inputs)

