import numpy as np

from core.activation_functions.activation_function import ActivationFunction


class ReLU(ActivationFunction):
    def execute(self, inputs: np.ndarray) -> np.ndarray:
        return np.maximum(0, inputs)

    def derivative(self, inputs: np.ndarray) -> np.ndarray:
        return np.where(inputs > 0, 1.0, 0.0)