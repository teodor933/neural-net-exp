from core.optimizers.optimizer import Optimizer


class SGD(Optimizer):
    def __init__(self, learning_rate: float = 1e-4):
        super().__init__(learning_rate)
        self.model = None

    def initialize(self, model):
        self.model = model

    def step(self):
        for weights, biases, dW, db in self.model.get_parameters():
            if dW is not None:
                weights -= self.lr * dW
            if db is not None:
                biases -= self.lr * db
