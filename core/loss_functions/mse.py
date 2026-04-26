import numpy as np

from core.loss_functions.loss_function import LossFunction


class MSE(LossFunction):
    def execute(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        return float(np.mean((predictions - targets) ** 2))

    def derivative(self, predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
        n = np.prod(targets.shape)
        return (2.0 / n) * (predictions - targets)