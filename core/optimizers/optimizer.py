from abc import ABC, abstractmethod


class Optimizer(ABC):
    def __init__(self, learning_rate: float = 1e-3):
        self.lr = learning_rate
        self.model = None

    @abstractmethod
    def initialize(self, model):
        raise NotImplementedError

    @abstractmethod
    def step(self):
        raise NotImplementedError
