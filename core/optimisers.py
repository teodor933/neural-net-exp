import numpy as np
from abc import ABC, abstractmethod


class Optimiser(ABC):
    @abstractmethod
    def update(self, weights, biases, dW, db):
        raise NotImplementedError

class SGD(Optimiser):
    def __init__(self, learning_rate=1e-4):
        self.lr = learning_rate

    def update(self, weights, biases, dW, db):
        weights -= self.lr * dW
        biases -= self.lr * db
        return weights, biases
