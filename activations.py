import numpy as np

class Linear: # Regression
    @staticmethod
    def execute(x):
        return x

    @staticmethod
    def derivative(x):
        return 1

class ReLU:
    @staticmethod
    def execute(x):
        return np.maximum(0, x)

    @staticmethod
    def derivative(x):
        return np.where(x > 0, 1, 0)

class LeakyReLU:
    def __init__(self, alpha=0.01):
        self.alpha = alpha

    def execute(self, x):
        return np.where(x > 0, x, self.alpha * x)

    def derivative(self, x):
        return np.where(x > 0, 1, self.alpha)

class Softmax: # Classification
    @staticmethod
    def execute(x):
        exp_x = np.exp(x - np.max(x))  # Stability trick to prevent overflow
        return exp_x / exp_x.sum(axis=0, keepdims=True)

    @staticmethod
    def derivative(x):
        s = Softmax.execute(x)
        return np.diagflat(s) - np.outer(s, s)
