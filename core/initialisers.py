import numpy as np
from abc import ABC, abstractmethod


class Initialiser(ABC):
    @abstractmethod
    def __call__(self, input_size: int, output_size: int) -> np.ndarray:
        raise NotImplementedError

class HeNormal(Initialiser):
    def __call__(self, input_size: int, output_size: int) -> np.ndarray:
        scale = np.sqrt(2.0 / input_size)
        return np.random.randn(output_size, input_size) * scale

class Normal(Initialiser):
    def __init__(self, scale=0.01):
        self.scale = scale

    def __call__(self, input_size: int, output_size: int) -> np.ndarray:
        return np.random.randn(output_size, input_size) * self.scale