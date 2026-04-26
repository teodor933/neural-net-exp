import numpy as np

from core.weight_initializers.weight_initializer import WeightInitializer


class GlorotNormal(WeightInitializer):
    def __call__(self, input_size: int, output_size: int) -> np.ndarray:
        scale = np.sqrt(2.0 / (input_size + output_size))
        return np.random.randn(input_size, output_size) * scale
