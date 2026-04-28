import numpy as np

from core.activation_functions.activation_function import ActivationFunction


class ELU(ActivationFunction):
    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha

    def execute(self, inputs: np.ndarray) -> np.ndarray:
        return np.where(inputs > 0, inputs, self.alpha * (np.exp(inputs) - 1))

    def derivative(self, inputs: np.ndarray) -> np.ndarray:
        return np.where(inputs > 0, 1.0, self.alpha * np.exp(inputs))
