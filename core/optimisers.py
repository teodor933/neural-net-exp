import numpy as np
from abc import ABC, abstractmethod


class Optimiser(ABC):
    def __init__(self, learning_rate: float = 1e-4):
        self.lr = learning_rate
        self.model = None

    @abstractmethod
    def initialise(self, model):
        raise NotImplementedError

    @abstractmethod
    def step(self):
        raise NotImplementedError

class SGD(Optimiser):
    def __init__(self, learning_rate: float = 1e-4):
        super().__init__(learning_rate)
        self.model = None

    def initialise(self, model):
        self.model = model

    def step(self):
        for weights, biases, dW, db in self.model.get_parameters():
            if dW is not None and db is not None:
                weights[:] -= self.lr * dW
                biases[:] -= self.lr * db


class SGDM(Optimiser):
    def __init__(self, learning_rate: float = 1e-4, gamma: float = 0.9):
        """
        Initialize the SGD with Momentum (SGDM) optimiser.

        Parameters:
            learning_rate (float): Step size for each parameter update (η).
            gamma (float): Momentum factor (γ), typically in the range [0.9, 0.99].

        The optimiser maintains per-layer "velocity" buffers (v_w and v_b)
        which accumulate gradients over time, weighted by the momentum factor.
        This helps accelerate learning in consistent gradient directions and
        reduces oscillation in noisy or curved loss surfaces.
        """
        super().__init__(learning_rate)
        self.gamma = gamma
        self.model = None
        self.velocities = None

    def initialise(self, model):
        self.model = model

    def step(self):
        """
        Performs a momentum-based parameter update.

        Parameters:
            weights (np.ndarray): Current weight matrix of a layer (θ_w).
            biases (np.ndarray): Current bias vector of a layer (θ_b).
            dW (np.ndarray): Gradient of the loss w.r.t. weights (∇J(θ_w)).
            db (np.ndarray): Gradient of the loss w.r.t. biases (∇J(θ_b)).

        This method updates weights and biases using momentum:

            v_t = (γ * v_{t-1}) + (η * ∇J(θ))
            θ = θ - v_t

        Where:
            - v_t is the current "velocity" (accumulated gradient),
            - γ is the momentum factor,
            - η is the learning rate.
        """

        if self.velocities is None:
            self.velocities = []
            for weights, biases, _, _, in self.model.get_parameters():
                v_w = np.zeros_like(weights)
                v_b = np.zeros_like(biases)
                self.velocities.append((v_w, v_b))

        for i, (weights, biases, dW, db) in enumerate(self.model.get_parameters()):
            if dW is None or db is None:
                continue

            v_w, v_b = self.velocities[i]

            # update in place
            v_w[:] = self.gamma * v_w + self.lr * dW
            v_b[:] = self.gamma * v_b + self.lr * db

            weights[:] -= v_w
            biases[:] -= v_b


class RMSProp(Optimiser):
    def __init__(self, learning_rate: float = 1e-4):
        super().__init__(learning_rate)
        self.params_with_grads = []

    def initialise(self, parameters):
        self.params_with_grads = list(parameters)

    def step(self):
        for weights, biases, dW, db in self.params_with_grads:
            pass