import numpy as np

from core.activation_functions.activation_function import ActivationFunction


class Sigmoid(ActivationFunction):
    def execute(self, inputs: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-inputs))

    def derivative(self, inputs: np.ndarray) -> np.ndarray:
        s = self.execute(inputs)
        return s * (1 - s)