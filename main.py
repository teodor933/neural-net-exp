import numpy as np

import matplotlib.pyplot as plt

from core.activation_functions.leaky_relu import LeakyReLU
from core.activation_functions.linear import Linear
from core.activation_functions.relu import ReLU
from core.layers.activation import Activation
from core.layers.dense import Dense
from core.loss_functions.mse import MSE
from core.models.neural_network import NeuralNetwork
from core.optimizers.sgd import SGD
from core.training.learner import Learner
from core.weight_initializers.glorot_normal import GlorotNormal
from core.weight_initializers.he_normal import HeNormal
from core.weight_initializers.normal import Normal


def main():
    np.random.seed(123)

    input_samples = np.linspace(-1, 1, 200).reshape(-1, 1)
    output_samples = 2*input_samples + 1

    model = NeuralNetwork(input_size=1)
    model.add_layer(Dense(16, weight_initializer=HeNormal()))
    model.add_layer(Activation(LeakyReLU(alpha=0.1)))
    model.add_layer(Dense(16, weight_initializer=HeNormal()))
    model.add_layer(Activation(LeakyReLU(alpha=0.1)))
    model.add_layer(Dense(1, weight_initializer=GlorotNormal()))
    model.add_layer(Activation(Linear()))

    model.summary()

    learner = Learner(
        model=model,
        loss_fn=MSE(),
        optimizer=SGD(learning_rate=5e-3, momentum=0.99)
    )

    history = learner.learn(
        input_samples=input_samples,
        output_samples=output_samples,
        epochs=500,
        batch_size=100,
        shuffle=True,
        verbose=True
    )

    plt.plot(history["loss"])
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.show()

    x_plot = np.linspace(-2.0, 2.0, 400).reshape(-1, 1)
    y_true = 2 * x_plot + 1
    y_pred = learner.predict(x_plot)

    plt.plot(x_plot, y_true, label="True function")
    plt.plot(x_plot, y_pred, label="Model prediction")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Model Predictions")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()



