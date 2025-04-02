import numpy as np

class MSELoss: # Regression
    @staticmethod
    def execute(y_pred, y_true):
        return np.mean((y_pred - y_true) ** 2)

    @staticmethod
    def derivative(y_pred, y_true):
        return (2 / y_true.shape[0]) * (y_pred - y_true)


class CrossEntropyLoss: # Classification, y_true should be one-hot encoded
    @staticmethod
    def execute(y_pred, y_true):
        epsilon = 1e-12
        y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
        return -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]

    @staticmethod
    def derivative(y_pred, y_true):
        return y_pred - y_true