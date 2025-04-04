import numpy as np

class Layer:
    def __init__(self, input_size, output_size, activation):
        self.weights = np.random.randn(output_size, input_size) * 0.01
        self.biases = np.zeros((output_size, 1))
        self.activation = activation

        self.inputs = None
        self.z = None
        self.outputs = None
        self.d_weights = None
        self.d_biases = None

    def forward(self, inputs):
        self.inputs = inputs
        self.z = np.dot(self.weights, self.inputs) + self.biases
        self.outputs = self.activation.execute(self.z)
        return self.outputs

    def backward(self, d_loss):
        d_z = d_loss * self.activation.derivative(self.z) # dL/dz = dL/da * da/dz
        self.d_weights = np.dot(d_z, self.inputs.T) / self.inputs.shape[1] # dL/dW = dL/dz * dz/dW
        self.d_biases = np.mean(d_z, axis=1, keepdims=True) # axis for rows instead
        d_inputs = np.dot(self.weights.T, d_z) # dL/dx = dL/dz * dz/dx
        return d_inputs, self.d_weights, self.d_biases

    def update(self, optimiser):
        self.weights, self.biases = optimiser.update(self.weights, self.biases,
                                                     self.d_weights, self.d_biases)