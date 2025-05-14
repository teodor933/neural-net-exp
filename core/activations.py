from abc import abstractmethod, ABC
import numpy as np

from core.initialisers import HeNormal, Normal

class Activation(ABC):
    @abstractmethod
    def execute(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def derivative(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

class Linear(Activation): # Regression
    def __init__(self):
        self.weight_initialiser = Normal(scale=0.01)

    def execute(self, x: np.ndarray) -> np.ndarray:
        return x

    def derivative(self, x: np.ndarray) -> np.ndarray:
        return np.ones_like(x)

class ReLU(Activation):
    def __init__(self):
        self.weight_initialiser = HeNormal()

    def execute(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, 1, 0)

class LeakyReLU(Activation):
    def __init__(self, alpha=0.01):
        self.alpha = alpha
        self.weight_initialiser = HeNormal()

    def execute(self, x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, x, self.alpha * x)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, 1, self.alpha)

class Tanh(Activation):
    def __init__(self):
        self.weight_initialiser = Normal(scale=0.01)

    def execute(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(x)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        return 1 - np.tanh(x) ** 2


class Softmax(Activation): # Classification
    def __init__(self):
        self.weight_initialiser = Normal(scale=0.01)

    def execute(self, x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x))  # Stability trick to prevent overflow
        return exp_x / exp_x.sum(axis=0, keepdims=True)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        s = Softmax.execute(self, x)
        return np.diagflat(s) - np.outer(s, s)
