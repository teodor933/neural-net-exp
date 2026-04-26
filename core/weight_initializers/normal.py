import numpy as np

from core.weight_initializers.weight_initializer import WeightInitializer


class Normal(WeightInitializer):
    def __init__(self, scale: float = 0.01) -> None:
        self.scale = scale

    def __call__(self, input_size: int, output_size: int) -> np.ndarray:
        return np.random.randn(input_size, output_size) * self.scale
