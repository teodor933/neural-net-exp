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


class SGDM(Optimiser):
    def __init__(self, learning_rate=1e-4, gamma=0.9):
        self.lr = learning_rate

        self.gamma = gamma
        self.v_w = None
        self.v_b = None

    def update(self, weights, biases, dW, db):
        """
        vt = γv_t−1 + η∇θJ(θ)
        θ = θ − vt
        """
        if self.v_w is None or self.v_b is None:
            self.v_w = np.zeros_like(weights)
            self.v_b = np.zeros_like(biases)

        self.v_w = self.gamma * self.v_w + self.lr * dW 
        self.v_b = self.gamma * self.v_b + self.lr * db
        
        weights -= self.v_w
        biases -= self.v_b
