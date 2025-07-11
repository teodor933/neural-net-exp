import numpy as np

from core.layers import Dense
from core.neural_network import NeuralNetwork
from core.activations import LeakyReLU, Linear, Tanh, ReLU
from core.loss_functions import MSELoss
from core.optimisers import SGD, SGDM
from learning.learner import Learner
from core.initialisers import HeNormal, Normal

import matplotlib.pyplot as plt

def main():
    np.random.seed(123)

    x_batch = np.random.uniform(-10, 10, (1, 1000))  # (1, 1000)
    y_batch = np.sin(x_batch)

    x_mean, x_std = np.mean(x_batch), np.std(x_batch)
    x_batch = ((x_batch - x_mean) / x_std)  # standard normal distribution z-score
    y_batch = y_batch

    nn = NeuralNetwork() # lazy config
    nn.add_layer(Dense(output_size=32, activation=LeakyReLU(alpha=0.01), weight_initialiser=HeNormal())) # lazy config
    nn.add_layer(Dense(output_size=16, activation=LeakyReLU(alpha=0.01), weight_initialiser=HeNormal())) # lazy config
    nn.add_layer(Dense(input_size=16, output_size=1, activation=Linear(), weight_initialiser=HeNormal())) # explicit config

    learning_rate = 0.005

    optimiser = SGDM(learning_rate=learning_rate, gamma=0.9)
    learner = Learner(model=nn,
                      loss_fn=MSELoss(),
                      optimiser=optimiser,
                      batch_size=50)

    epochs = 4000

    training_loss = learner.learn(x_batch, y_batch,
                                  epochs=epochs,
                                  verbose=True)

    x_eval = np.linspace(-10, 10, 1000).reshape(1, -1)  # Shape: (1, 1000)
    x_eval_normalized = (x_eval - x_mean) / x_std
    y_true = np.sin(x_eval)
    y_pred = nn.predict(x_eval_normalized)

    # Plotting
    plt.plot(x_eval.flatten(), y_true.flatten(), label="True sin(x)", color='blue')
    plt.plot(x_eval.flatten(), y_pred.flatten(), label="NN Prediction", color='red')
    plt.legend()
    plt.title("Neural Network Approximation of sin(x)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()



