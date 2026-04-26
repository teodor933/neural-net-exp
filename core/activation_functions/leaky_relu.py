import numpy as np

from core.activation_functions.activation_function import ActivationFunction


class LeakyReLU(ActivationFunction):
    def __init__(self, alpha=0.01):
        self.alpha = alpha

    def execute(self, inputs: np.ndarray) -> np.ndarray:
        return np.where(inputs > 0, inputs, self.alpha * inputs)

    def derivative(self, inputs: np.ndarray) -> np.ndarray:
        return np.where(inputs > 0, 1.0, self.alpha)