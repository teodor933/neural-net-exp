from abc import ABC, abstractmethod

from core.models.neural_network import NeuralNetwork


class Optimizer(ABC):
    def __init__(self, learning_rate: float = 1e-3) -> None:
        self.lr = learning_rate
        self.model: NeuralNetwork | None = None

    def initialize(self, model: NeuralNetwork) -> None:
        self.model = model

    @abstractmethod
    def step(self) -> None:
        raise NotImplementedError
