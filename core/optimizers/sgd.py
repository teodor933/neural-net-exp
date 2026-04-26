import numpy as np

from core.optimizers.optimizer import Optimizer


class SGD(Optimizer):
    def __init__(
            self,
            learning_rate: float = 1e-4,
            momentum: float = 0.9,
            weight_decay: float = 0.0,
    ) -> None:
        super().__init__(learning_rate)
        self.momentum = momentum
        self.weight_decay = weight_decay

    def step(self):
        if self.model is None:
            raise RuntimeError("Optimizer has not been initialized with a model.")

        for parameter in self.model.get_parameters():
            if not parameter.trainable:
                continue

            if parameter.gradient is None:
                continue

            gradient = parameter.gradient

            if self.weight_decay != 0.0:
                gradient = gradient + self.weight_decay * parameter.data

            if self.momentum != 0.0:
                velocity = parameter.state.get("velocity")

                if velocity is None:
                    velocity = np.zeros_like(parameter.data)

                velocity = self.momentum * velocity - self.lr * gradient

                parameter.state["velocity"] = velocity
                parameter.data = parameter.data + velocity
            else:
                parameter.data = parameter.data - self.lr * gradient
