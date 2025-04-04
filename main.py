import numpy as np

from core.neural_network import NeuralNetwork
from core.activations import LeakyReLU, Linear
from core.loss_functions import MSELoss
from core.optimisers import SGD
from learning.learner import Learner

def main():
    x_batch = np.random.uniform(-6, 6, (1, 1000))  # (1, 1000)
    y_batch = np.sin(x_batch)

    learning_rate = 1e-2

    nn = NeuralNetwork(input_size=x_batch.shape[0])
    nn.add_layer(64, LeakyReLU(alpha=0.01))
    nn.add_layer(32, LeakyReLU(alpha=0.01))
    nn.add_layer(1, Linear())

    optimiser = SGD(learning_rate=learning_rate)
    learner = Learner(model=nn,
                      loss_fn=MSELoss,
                      optimiser=optimiser,
                      batch_size=128)

    epochs = 10000

    training_loss = learner.learn(x_batch, y_batch,
                                  epochs=epochs,
                                  verbose=True)

if __name__ == "__main__":
    main()


