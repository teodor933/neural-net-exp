import numpy as np

from neural_network import NeuralNetwork
from activations import LeakyReLU, Linear
from loss_functions import MSELoss
from optimisers import SGD

np.random.seed(123)

batch_size = 64
x_batch = np.random.uniform(-6, 6, (1, batch_size))  # (1, 64)
y_batch = np.sin(x_batch)

learning_rate = 1e-2
optimiser = SGD(learning_rate=learning_rate)
nn = NeuralNetwork(input_size=x_batch.shape[0], optimiser=optimiser, batch_size=batch_size)

nn.add_layer(128, LeakyReLU(alpha=0.01))
nn.add_layer(64, LeakyReLU(alpha=0.01))
nn.add_layer(1, Linear())

epochs = 200000
print_frequency = 100

for epoch in range(epochs):
    loss = nn.compute_loss(x_batch, y_batch, MSELoss)
    nn.backpropagate(x=x_batch, y_true=y_batch, loss_fn=MSELoss)
    if epoch % print_frequency == 0:
        print(f"Epoch: {epoch}/{epochs}, Loss: {loss:.6f}")

print(f"Final loss: {nn.compute_loss(x_batch, y_batch, MSELoss)}")
