from typing import Optional

import numpy as np
from abc import ABC, abstractmethod


class Optimiser(ABC):
    @abstractmethod
    def update(self, weights: np.ndarray, biases: np.ndarray,
               dW: np.ndarray, db: np.ndarray):
        raise NotImplementedError

class SGD(Optimiser):
    def __init__(self, learning_rate: float = 1e-4):
        self.lr = learning_rate

    def update(self, weights: np.ndarray, biases: np.ndarray,
               dW: np.ndarray, db: np.ndarray):
        weights -= self.lr * dW
        biases -= self.lr * db


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
        self.lr = learning_rate
        self.gamma = gamma

        self.v_w = {}
        self.v_b = {}

    def update(self, weights: np.ndarray, biases: np.ndarray,
               dW: np.ndarray, db: np.ndarray):
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
        weights_id = id(weights)
        biases_id = id(biases)

        if weights_id not in self.v_w:
            self.v_w[weights_id] = np.zeros_like(weights)
            self.v_b[biases_id] = np.zeros_like(biases)

        # [:] for mutating the existing array instead of rebinding, updating data in place
        self.v_w[weights_id][:] = self.gamma * self.v_w[weights_id] + self.lr * dW
        self.v_b[biases_id][:] = self.gamma * self.v_b[biases_id] + self.lr * db
        
        weights -= self.v_w[weights_id]
        biases -= self.v_b[biases_id]


class RMSProp(Optimiser):
    def __init__(self):
        pass

    def update(self, weights: np.ndarray, biases: np.ndarray,
               dW: np.ndarray, db: np.ndarray):
        pass