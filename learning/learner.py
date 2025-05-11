import numpy as np

class Learner:
    def __init__(self, model, loss_fn, optimiser, batch_size=1):
        self.model = model
        self.loss_fn = loss_fn
        self.optimiser = optimiser
        self.batch_size = batch_size

    def learn_step(self, input_batch, output_batch):
        pred, _ = self.model.forward(input_batch) # get model output

        loss = self.loss_fn.execute(pred, output_batch) # get loss

        d_loss = self.loss_fn.derivative(pred, output_batch) # derivative of the loss
        self.model.backpropagate(d_loss, self.optimiser) # iterate network

        return loss # other use cases

    def learn(self, input_batch, output_batch, epochs,
              verbose=True):
        num_samples = input_batch.shape[1]
        for epoch in range(epochs):
            # shuffle and randomise the data to be used, helps with overfitting
            permutation = np.random.permutation(num_samples)
            x_shuffled = input_batch[:, permutation]
            y_shuffled = output_batch[:, permutation]

            for i in range(0, num_samples, self.batch_size):
                x_batch = x_shuffled[:, i:i+self.batch_size]
                y_batch = y_shuffled[:, i:i+self.batch_size]
                self.learn_step(x_batch, y_batch)

            if verbose and epoch % (epochs // 10 or 1) == 0:
                loss = self.loss_fn.execute(self.model.forward(x_shuffled)[0], y_shuffled) # index 0 is output, 1 is act function
                print(f"Epoch {epoch}/{epochs}, Loss: {loss:.6f}")

        return self