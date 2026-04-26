from abc import ABC, abstractmethod

import numpy as np


class WeightInitializer(ABC):
    @abstractmethod
    def __call__(self, input_size: int, output_size: int) -> np.ndarray:
        raise NotImplementedError
