import numpy as np

from core.activation_functions.activation_function import ActivationFunction


class Tanh(ActivationFunction):
    def execute(self, inputs: np.ndarray) -> np.ndarray:
        return np.tanh(inputs)

    def derivative(self, inputs: np.ndarray) -> np.ndarray:
        return 1 - np.tanh(inputs) ** 2