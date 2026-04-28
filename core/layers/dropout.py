import numpy as np

from core.layers.layer import Layer


class Dropout(Layer):
    def __init__(self, rate: float = 0.5) -> None:
        super().__init__()

        if not 0.0 <= rate < 1.0:
            raise ValueError("Dropout rate must be in [0.0, 1.0).")

        self.rate = rate
        self.mask: np.ndarray | None = None

    def build(self, input_size: int) -> None:
        self.input_size = input_size
        self.output_size = input_size
        self.built = True

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        if not self.built:
            self.build(inputs.shape[1])

        self.inputs = inputs

        if not self.training or self.rate == 0.0:
            self.mask = None
            self.outputs = inputs
            return self.outputs

        keep_probability = 1.0 - self.rate

        # Scale mask by (1/p) to maintain roughly the same signal
        self.mask = (np.random.rand(*inputs.shape) < keep_probability).astype(np.float32) / keep_probability

        self.outputs = inputs * self.mask
        return self.outputs

    def backward(self, d_loss: np.ndarray) -> np.ndarray:
        if self.inputs is None:
            raise RuntimeError("Cannot call backward() on Dropout before forward().")

        if not self.training or self.mask is None:
            return d_loss

        return d_loss * self.mask