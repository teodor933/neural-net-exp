import numpy as np
from abc import ABC, abstractmethod


class Loss(ABC):
    @abstractmethod
    def execute(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        raise NotImplementedError

    @abstractmethod
    def derivative(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        raise NotImplementedError

class MSELoss(Loss): # Regression
    def execute(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        return float(np.mean((y_pred - y_true) ** 2))

    def derivative(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        n = np.prod(y_true.shape)
        return (2 / n) * (y_pred - y_true) # features * samples to get proper mean if you ever ask why division by n

class CrossEntropyLoss(Loss): # Classification, y_true should be one-hot encoded
    def execute(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        epsilon = 1e-12
        y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
        return -np.sum(y_true * np.log(y_pred)) / y_true.shape[1]

    def derivative(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        return (y_pred - y_true) / y_true.shape[1]
