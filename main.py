import numpy as np

from core.neural_network import NeuralNetwork
from core.activations import LeakyReLU, Linear
from core.loss_functions import MSELoss
from core.optimisers import SGD
from learning.learner import Learner
from core.initialisers import HeNormal, Normal

import matplotlib.pyplot as plt

def main():
    np.random.seed(123)
    
    x_batch = np.random.uniform(-6, 6, (1, 1000))  # (1, 1000)
    y_batch = np.sin(x_batch)

    learning_rate = 0.1

    nn = NeuralNetwork(input_size=x_batch.shape[0])
    nn.add_layer(64, LeakyReLU(alpha=0.01), weight_initialiser=HeNormal())
    nn.add_layer(32, LeakyReLU(alpha=0.01), weight_initialiser=HeNormal())
    nn.add_layer(1, Linear())

    optimiser = SGD(learning_rate=learning_rate)
    learner = Learner(model=nn,
                      loss_fn=MSELoss(),
                      optimiser=optimiser,
                      batch_size=128)

    epochs = 10000

    training_loss = learner.learn(x_batch, y_batch,
                                  epochs=epochs,
                                  verbose=True)

    x_eval = np.linspace(-6, 6, 1000).reshape(1, -1)  # Shape: (1, 1000)
    y_true = np.sin(x_eval)
    y_pred = nn.forward(x_eval)

    # Plotting
    plt.plot(x_eval.flatten(), y_true.flatten(), label="True sin(x)", color='blue')
    plt.plot(x_eval.flatten(), y_pred[0].flatten(), label="NN Prediction", color='red')
    plt.legend()
    plt.title("Neural Network Approximation of sin(x)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()


