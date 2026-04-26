from abc import ABC, abstractmethod

import numpy as np


class LossFunction(ABC):
    @abstractmethod
    def execute(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        raise NotImplementedError

    @abstractmethod
    def derivative(self, predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
        raise NotImplementedError