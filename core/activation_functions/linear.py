import numpy as np

from core.activation_functions.activation_function import ActivationFunction


class Linear(ActivationFunction):
    def execute(self, inputs: np.ndarray) -> np.ndarray:
        return inputs

    def derivative(self, inputs: np.ndarray) -> np.ndarray:
        return np.ones_like(inputs)