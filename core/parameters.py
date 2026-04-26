from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Parameter:
    data: np.ndarray
    name: str = ""
    trainable: bool = True
    gradient: np.ndarray | None = None
    state: dict[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self.data = np.asarray(self.data, dtype=np.float32)

        if self.gradient is not None:
            self.gradient = np.asarray(self.gradient, dtype=np.float32)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    def zero_gradient(self) -> None:
        self.gradient = None

    def set_gradient(self, gradient: np.ndarray) -> None:
        self.gradient = np.asarray(gradient, dtype=np.float32)
