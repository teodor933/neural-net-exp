import numpy as np

from neural_network import *
from activations import *
from loss_functions import *
from optimisers import *

target_output = np.array([0.1, 0.2, 0.3, 0.4, 0.5, -0.5, -0.4, -0.3, -0.2, -0.1])

nn = NeuralNetwork(input_size=len(target_output), optimiser=SGD(learning_rate=1e-4))
nn.add_layer(128, LeakyReLU(alpha=0.01))
nn.add_layer(64, LeakyReLU(alpha=0.01))
nn.add_layer(10, Linear)

for i in range(100):
    print(f"OUTPUT: {nn.forward(target_output)}, LOSS: {nn.compute_loss(target_output, target_output, MSELoss)}")
    nn.backpropagate(x=target_output, y_true=target_output, loss_fn=MSELoss)

output = nn.forward(target_output)
loss = nn.compute_loss(target_output, target_output, MSELoss)
print("Predicted Output:", output)
print("Loss (MSE):", loss)

